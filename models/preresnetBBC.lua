--
--  Copyright (c) 2016, Facebook, Inc.
--  All rights reserved.
--
--  This source code is licensed under the BSD-style license found in the
--  LICENSE file in the root directory of this source tree. An additional grant
--  of patent rights can be found in the PATENTS file in the same directory.
--
--  The full pre-activation ResNet variation from the technical report
-- "Identity Mappings in Deep Residual Networks" (http://arxiv.org/abs/1603.05027)
--

local nn = require 'nn'
require 'cunn'

local Convolution = cudnn.SpatialConvolution
local TemporalConvolution = cudnn.TemporalConvolution
local Avg = cudnn.SpatialAveragePooling
local ReLU = nn.ReLU
local TMax = nn.TemporalMaxPooling
local TAvg = nn.TemporalAveragePooling
local Max = nn.SpatialMaxPooling
local VMax = cudnn.VolumetricMaxPooling
local SBatchNorm = nn.SpatialBatchNormalization
local VConvolution = cudnn.VolumetricConvolution
local VBatchNorm = cudnn.VolumetricBatchNormalization
local BatchNorm = nn.BatchNormalization

local function createModel(opt)
   local depth = opt.depth
   local shortcutType = opt.shortcutType or 'B'
   local iChannels

   -- The shortcut layer is either identity or 1x1 convolution
   local function shortcut(nInputPlane, nOutputPlane, stride)
      local useConv = shortcutType == 'C' or
         (shortcutType == 'B' and nInputPlane ~= nOutputPlane)
      if useConv then
         -- 1x1 convolution
         return nn.Sequential()
            :add(Convolution(nInputPlane, nOutputPlane, 1, 1, stride, stride))
      elseif nInputPlane ~= nOutputPlane then
         -- Strided, zero-padded identity shortcut
         return nn.Sequential()
            :add(nn.SpatialAveragePooling(1, 1, stride, stride))
            :add(nn.Concat(2)
               :add(nn.Identity())
               :add(nn.MulConstant(0)))
      else
         return nn.Identity()
      end
   end

   -- Typically shareGradInput uses the same gradInput storage for all modules
   -- of the same type. This is incorrect for some SpatialBatchNormalization
   -- modules in this network b/c of the in-place CAddTable. This marks the
   -- module so that it's shared only with other modules with the same key
   local function ShareGradInput(module, key)
      assert(key)
      module.__shareGradInputKey = key
      return module
   end

   -- The basic residual layer block for 18 and 34 layer network, and the
   -- CIFAR networks
   local function basicblock(n, stride, type)
      local nInputPlane = iChannels
      iChannels = n
      local block = nn.Sequential()
      local s = nn.Sequential()
      if type == 'both_preact' then
         block:add(ShareGradInput(SBatchNorm(nInputPlane), 'preact'))
         block:add(ReLU(true))
      elseif type ~= 'no_preact' then
         s:add(SBatchNorm(nInputPlane))
         s:add(ReLU(true))
      end
      s:add(Convolution(nInputPlane,n,3,3,stride,stride,1,1))
      s:add(SBatchNorm(n))
      s:add(ReLU(true))
      s:add(Convolution(n,n,3,3,1,1,1,1))
      return block
         :add(nn.ConcatTable()
            :add(s)
            :add(shortcut(nInputPlane, n, stride)))
         :add(nn.CAddTable(true))
   end

   -- The bottleneck residual layer for 50, 101, and 152 layer networks
   local function bottleneck(n, stride, type)
      local nInputPlane = iChannels
      iChannels = n * 4
      local block = nn.Sequential()
      local s = nn.Sequential()
      if type == 'both_preact' then
         block:add(ShareGradInput(SBatchNorm(nInputPlane), 'preact'))
         block:add(ReLU(true))
      elseif type ~= 'no_preact' then
         s:add(SBatchNorm(nInputPlane))
         s:add(ReLU(true))
      end
      s:add(Convolution(nInputPlane,n,1,1,1,1,0,0))
      s:add(SBatchNorm(n))
      s:add(ReLU(true))
      s:add(Convolution(n,n,3,3,stride,stride,1,1))
      s:add(SBatchNorm(n))
      s:add(ReLU(true))
      s:add(Convolution(n,n*4,1,1,1,1,0,0))
      return block
         :add(nn.ConcatTable()
            :add(s)
            :add(shortcut(nInputPlane, n * 4, stride)))
         :add(nn.CAddTable(true))
   end

   -- Creates count residual blocks with specified number of features
   local function layer(block, features, count, stride, type)
      local s = nn.Sequential()
      if count < 1 then
        return s
      end
      s:add(block(features, stride,
                  type == 'first' and 'no_preact' or 'both_preact'))
      for i=2,count do
         s:add(block(features, 1))
      end
      return s
   end
   
   local function create_resnet(featDim)     
   
      local smodel = nn.Sequential()
      local def, nFeatures, block
      if opt.dataset == 'BBCnet' then
      -- Configurations for ResNet:
      --  num. residual blocks, num features, residual block function
         local cfg = {
            [18]  = {{2, 2, 2, 2}, 512, basicblock},
            [34]  = {{3, 4, 6, 3}, 512, basicblock},
            [50]  = {{3, 4, 6, 3}, 2048, bottleneck},
            [101] = {{3, 4, 23, 3}, 2048, bottleneck},
            [152] = {{3, 8, 36, 3}, 2048, bottleneck},
            [200] = {{3, 24, 36, 3}, 2048, bottleneck},
            }

         assert(cfg[depth], 'Invalid depth: ' .. tostring(depth))
         def, nFeatures, block = table.unpack(cfg[depth])
         iChannels = 64
         print(' | ResNet-' .. depth .. ' BBCnet')
         -- The ResNet ImageNet model

         smodel:add(layer(block, 64,  def[1], 1, 'first'))
         smodel:add(layer(block, 128, def[2], 2))
         smodel:add(layer(block, 256, def[3], 2))
         smodel:add(layer(block, 512, def[4], 2))
         smodel:add(ShareGradInput(SBatchNorm(iChannels), 'last'))
         smodel:add(ReLU(true))
         smodel:add(Avg(4, 4, 1, 1))
         smodel:add(nn.View(-1):setNumInputDims(3))
         smodel:add(nn.Linear(nFeatures, featDim))
         smodel:add(BatchNorm(featDim))

      else
         error('invalid dataset: ' .. opt.dataset)
      end
      return smodel, nFeatures
   end
   
   local model = nn.Sequential()
   model:add(nn.Transpose({2,3}))
   model:add(nn.View(-1,64,28,28):setNumInputDims(5))
   local smodel, nFeatures = create_resnet(opt.inputDim)
   model:add(smodel) 
   model:add(nn.View(opt.batchSize/opt.nGPU,-1,opt.inputDim):setNumInputDims(2))
      
   local function ConvInit(name)
      for k,v in pairs(model:findModules(name)) do
         local n = v.kW*v.kH*v.nOutputPlane
         v.weight:normal(0,math.sqrt(2/n))
         if cudnn.version >= 4000 then
            v.bias = nil
            v.gradBias = nil
         else
            v.bias:zero()
         end
      end
   end
   local function BNInit(name)
      for k,v in pairs(model:findModules(name)) do
         v.weight:fill(1)
         v.bias:zero()
      end
   end
   local function ConvInitV(name)
      for k,v in pairs(model:findModules(name)) do
         local n = v.kT*v.kW*v.kH*v.nOutputPlane
         v.weight:normal(0,math.sqrt(2/n))
         if cudnn.version >= 4000 then
            v.bias = nil
            v.gradBias = nil
         else
            v.bias:zero()
         end
      end
   end
   
   ConvInit('cudnn.SpatialConvolution')
   ConvInit('cudnn.TemporalConvolution')
   ConvInit('nn.SpatialConvolution')
   ConvInitV('cudnn.VolumetricConvolution')
   BNInit('fbnn.SpatialBatchNormalization')
   BNInit('cudnn.SpatialBatchNormalization')
   BNInit('nn.SpatialBatchNormalization')
   BNInit('cudnn.VolumetricBatchNormalization')
   BNInit('nn.BatchNormalization')
   
   for k,v in pairs(model:findModules('nn.Linear')) do
      v.bias:zero()
   end
   
   model:cuda()

   if opt.cudnn == 'deterministic' then
      model:apply(function(m)
         if m.setMode then m:setMode(1,1,1) end
      end)
   end

   return model
end

return createModel
