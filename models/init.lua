--
--  Copyright (c) 2016, Facebook, Inc.
--  All rights reserved.
--
--  This source code is licensed under the BSD-style license found in the
--  LICENSE file in the root directory of this source tree. An additional grant
--  of patent rights can be found in the PATENTS file in the same directory.
--
--  Generic model creating code. For the specific ResNet model see
--  models/resnet.lua
--

require 'nn'
require 'cunn'
require 'cudnn'
require 'rnn'
local NoBackprop = require 'NoBackprop'
local M = {}

function M.setup(opt, checkpoint)
   local model
   if checkpoint then
      local modelPath = paths.concat(opt.resume, checkpoint.modelFile)
      assert(paths.filep(modelPath), 'Saved model not found: ' .. modelPath)
      print('=> Resuming model from ' .. modelPath)
      model = torch.load(modelPath):cuda()
   elseif opt.retrain ~= 'none' then
      local model_init = torch.load(opt.retrain)
      assert(paths.filep(opt.retrain), 'File not found: ' .. opt.retrain)
      print('Loading model from file: ' .. opt.retrain)

      if opt.netType ==	'LSTM_init' then
      -- LSTM initialization
      -- This is for training the LSTM, given a net trained with temp_conv backend. It also places frontend and ResNet in NoBackpropers, to freeze their parameters. Set weightDecay = 0.0 in this case.

         opt.weightDecay = 0.0
         model_init:remove(3) -- remove temp-conv backend
         local backend = require('models/backend_LSTM')(opt)
         model = nn.Sequential():add(nn.NoBackprop(model_init:get(1))):add(nn.NoBackprop(model_init:get(2))):add(backend):cuda()
         print('TempConv Replaced by LSTM. NoBackprob containers added.')       
      elseif opt.netType == 'LSTM' then  
      -- End-to-end training 
      -- This is for training the net with LSTM end-to-end, assuming that the LSTM has been initialized using the code above. It removes the NoBackprop containers.      
         model = nn.Sequential():add(model_init:get(1):get(1)):add(model_init:get(2):get(1)):add(model_init:get(3)):cuda()
         print('NoBackprob containers removed for E2E training.')
      end
      
   else
      -- requires editing
      print('=> Creating model')
      local frontend = require('models/frontend3D')(opt)
      local resnet = require('models/preresnetBBC')(opt)
      local backend
      if opt.netType == 'temp_conv' then
         backend = require('models/backend_conv')(opt)
      else
         backend = require('models/backend_LSTM')(opt) -- I haven't succeeded to train the net from scratch with LSTM backend.
      end
      model = nn.Sequential():add(frontend):add(resnet):add(backend):cuda()
   end

   -- First remove any DataParallelTable
   if torch.type(model) == 'nn.DataParallelTable' then
      model = model:get(1)
   end
   
   -- This is the only module where batchsize is hardcoded. So update is required.

   -- optnet is an general library for reducing memory usage in neural networks
   if opt.optnet then
      local optnet = require 'optnet'
      local imsize = opt.dataset == 'imagenet' and 224 or 32
      local sampleInput = torch.zeros(4,3,imsize,imsize):cuda()
      optnet.optimizeMemory(model, sampleInput, {inplace = false, mode = 'training'})
   end

   -- This is useful for fitting ResNet-50 on 4 GPUs, but requires that all
   -- containers override backwards to call backwards recursively on submodules
   if opt.shareGradInput then
      M.shareGradInput(model)
   end

   -- For resetting the classifier when fine-tuning on a different Dataset
   if opt.resetClassifier and not checkpoint then
      print(' => Replacing classifier with ' .. opt.nClasses .. '-way classifier')

      local orig = model:get(#model.modules)
      assert(torch.type(orig) == 'nn.Linear',
         'expected last layer to be fully connected')

      local linear = nn.Linear(orig.weight:size(2), opt.nClasses)
      linear.bias:zero()

      model:remove(#model.modules)
      model:add(linear:cuda())
   end

   -- Set the CUDNN flags
   if opt.cudnn == 'fastest' then
      cudnn.fastest = true
      cudnn.benchmark = true
   elseif opt.cudnn == 'deterministic' then
      -- Use a deterministic convolution implementation
      model:apply(function(m)
         if m.setMode then m:setMode(1, 1, 1) end
      end)
   end

   -- Wrap the model with DataParallelTable, if using more than one GPU
   if opt.nGPU > 1 then
      local gpus = torch.range(1, opt.nGPU):totable()
      local fastest, benchmark = cudnn.fastest, cudnn.benchmark

      local dpt = nn.DataParallelTable(1, true, true)
         :add(model, gpus)
         :threads(function()
            local cudnn = require 'cudnn'
            local rnn = require 'rnn' --required for LSTMs
            cudnn.fastest, cudnn.benchmark = fastest, benchmark
         end)
      dpt.gradInput = nil
      model = dpt:cuda()
   end
   
   local criterion

   if opt.netType == "temp_conv" then
       criterion = nn.CrossEntropyCriterion():cuda() -- for temp-conv backend
   else 
       criterion = nn.SequencerCriterion(nn.ClassNLLCriterion():cuda()):cuda() -- for LSTM backend
   end
   
   return model, criterion
end

function M.shareGradInput(model)
   local function sharingKey(m)
      local key = torch.type(m)
      if m.__shareGradInputKey then
         key = key .. ':' .. m.__shareGradInputKey
      end
      return key
   end

   -- Share gradInput for memory efficient backprop
   local cache = {}
   model:apply(function(m)
      local moduleType = torch.type(m)
      if torch.isTensor(m.gradInput) and moduleType ~= 'nn.ConcatTable' then
         local key = sharingKey(m)
         if cache[key] == nil then
            cache[key] = torch.CudaStorage(1)
         end
         m.gradInput = torch.CudaTensor(cache[key], 1, 0)
      end
   end)
   for i, m in ipairs(model:findModules('nn.ConcatTable')) do
      if cache[i % 2] == nil then
         cache[i % 2] = torch.CudaStorage(1)
      end
      m.gradInput = torch.CudaTensor(cache[i % 2], 1, 0)
   end
end

return M
