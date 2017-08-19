--
--  Copyright (c) 2016, Facebook, Inc.
--  All rights reserved.
--
--  This source code is licensed under the BSD-style license found in the
--  LICENSE file in the root directory of this source tree. An additional grant
--  of patent rights can be found in the PATENTS file in the same directory.
--
--  The training loop and learning rate schedule
--

local optim = require 'optim'

local M = {}
local Trainer = torch.class('LSTM.Trainer', M)

function Trainer:__init(model, criterion, opt, optimState)
   self.model = model
   self.criterion = criterion
   self.optimState = optimState or {
      learningRate = opt.LR,
      learningRateDecay = 0.0,
      momentum = opt.momentum,
      nesterov = true,
      dampening = 0.0,
      weightDecay = opt.weightDecay, --SOS use 0.0 in this experiment to leave fronetend and ResNet unaltered
   }
   self.opt = opt
   self.params, self.gradParams = model:getParameters()
end

function Trainer:runningmean(xn,T)
   table.remove(T,1)
   table.insert(T,xn)
   local m = torch.Tensor(T):mean()
   return m,T
end

function Trainer:train(epoch, dataloader)
   -- Trains the model for a single epoch
   self.optimState.learningRate = self:learningRate(epoch)

   local timer = torch.Timer()
   local dataTimer = torch.Timer()

   local function feval()
      return self.criterion.output, self.gradParams
   end

   local trainSize = dataloader:size()
   local top1Sum, top5Sum, lossSum = 0.0, 0.0, 0.0
   local N = 0

   print('=> Training epoch # ' .. epoch)
   -- set the batch norm to training mode
   self.model:training()

   local nmean = 100
   local Terr = torch.FloatTensor(nmean):fill(0):totable()
   local Ttop1 = torch.FloatTensor(nmean):fill(0):totable()    
   local rerr = 0
   local rtop1 = 0
   local fake_examples = torch.IntTensor()

   for n, sample in dataloader:run() do
      local dataTime = dataTimer:time().real
      local loss = 0.0
      local top1 = 0.0
      local top5 = 0.0
      fake_examples = self:copyInputs(sample)
     
       if fake_examples:sum() == 0 then
          
          local output = self.model:forward(self.input):float()
          local batchSize = output:size(1)
          local loss = self.criterion:forward(self.model.output, self.target)
          self.model:zeroGradParameters()
          self.criterion:backward(self.model.output, self.target)
          self.model:backward(self.input, self.criterion.gradInput)

          optim.adam(feval, self.params, self.optimState) -- I use adam in my latest experiments

          local top1, top5 = self:computeScore(output, sample.target, 1)
          top1Sum = top1Sum + top1*batchSize
          top5Sum = top5Sum + top5*batchSize
          lossSum = lossSum + loss*batchSize
          N = N + batchSize
          rerr, Terr = self:runningmean(loss,Terr)
          rtop1, Ttop1 = self:runningmean(top1,Ttop1)

          print((' | Epoch: [%d][%d/%d]    Time %.3f  Data %.3f  Err %1.4f (%1.4f) top1 %7.3f (%7.3f)  top5 %7.3f'):format(
          epoch, n, trainSize, timer:time().real, dataTime, loss, rerr, top1, rtop1, top5))

          -- check that the storage didn't get changed do to an unfortunate getParameters call
          assert(self.params:storage() == self.model:parameters()[1]:storage())

          timer:reset()
          dataTimer:reset()
      end
   end

   return top1Sum / N, top5Sum / N, lossSum / N
end

function Trainer:test(epoch, dataloader)
   -- Computes the top-1 and top-5 err on the validation set

   local timer = torch.Timer()
   local dataTimer = torch.Timer()
   local size = dataloader:size()

   local nCrops = self.opt.tenCrop and 10 or 1
   local top1Sum, top5Sum, top10Sum = 0.0, 0.0, 0.0
   local N = 0
   
   self.model:evaluate()
   local fake_examples = torch.IntTensor()

   for n, sample in dataloader:run() do
      local dataTime = dataTimer:time().real
      -- Copy input and target to the GPU
      fake_examples = self:copyInputs(sample)

      local output = self.model:forward(self.input):float()
      local batchSize = output:size(1)/nCrops - fake_examples:sum()
      local loss = self.criterion:forward(self.model.output, self.target)
      local top1, top5, top10 = self:computeScore(output[{{1,batchSize}}], sample.target[{{1,batchSize}}], nCrops)
      top1Sum = top1Sum + top1*batchSize
      top5Sum = top5Sum + top5*batchSize
      top10Sum = top10Sum + top10*batchSize
      N = N + batchSize

      print((' | Test: [%d][%d/%d]    Time %.3f  Data %.3f  top1 %7.3f (%7.3f)  top5 %7.3f (%7.3f)  top10 %7.3f (%7.3f)'):format(
         epoch, n, size, timer:time().real, dataTime, top1, top1Sum / N, top5, top5Sum / N, top10, top10Sum / N))

      timer:reset()
      dataTimer:reset()
   end
   self.model:training()

   print((' * Finished epoch # %d     top1: %7.3f  top5: %7.3f  top10: %7.3f\n'):format(
      epoch, top1Sum / N, top5Sum / N, top10Sum / N))

   return top1Sum / N, top5Sum / N, top10Sum / N
end

function Trainer:computeScore(output, target, nCrops)
   if output:dim() == 3 then
       output = torch.sum(output,2):squeeze(2) -- sum over all outputs (for LSTM, that has one logsoftmax per frame)
   end

   if nCrops > 1 then
      -- Sum over crops
      output = output:view(output:size(1) / nCrops, nCrops, output:size(2))
         --:exp()
         :sum(2):squeeze(2)
   end

   -- Computes the top1 and top5 error rate
   local batchSize = output:size(1)

   local _ , predictions = output:float():sort(2, true) -- descending

   -- Find which predictions match the target
   local correct = predictions:eq(
      target:long():view(batchSize, 1):expandAs(output))

   -- Top-1 score
   local top1 = 1.0 - (correct:narrow(2, 1, 1):sum() / batchSize)

   -- Top-5 score, if there are at least 5 classes
   local len = math.min(5, correct:size(2))
   local top5 = 1.0 - (correct:narrow(2, 1, len):sum() / batchSize)

-- Top-10 score, if there are at least 10 classes
   local len = math.min(10, correct:size(2))
   local top10 = 1.0 - (correct:narrow(2, 1, len):sum() / batchSize)

   return top1 * 100, top5 * 100, top10 * 100
end

function Trainer:copyInputs(sample)
   -- To have a fixed number of examples per minibatch I add some fake examples in the last minibatch of each epoch, which are not used in training or in testing

   self.input = self.input or (self.opt.nGPU == 1
      and torch.CudaTensor()
      or torch.CudaTensor())

   self.target = self.target or (torch.CudaLongTensor and torch.CudaLongTensor() or torch.CudaTensor())
   local fake_examples = torch.IntTensor(self.opt.batchSize):fill(0)
   if sample.input:size(1)<self.opt.batchSize then
      fake_examples[{{sample.input:size(1)+1,-1}}]:fill(1)
      --video:
      local P = torch.FloatTensor(self.opt.batchSize - sample.input:size(1),sample.input:size(2),sample.input:size(3),sample.input:size(4),sample.input:size(5)):normal(0,1)
      sample.input = torch.cat(sample.input,P:typeAs(sample.input),1)
      --target:
      local O = torch.LongTensor(self.opt.batchSize - sample.target:size(1)):random(1,500)
      sample.target = torch.cat(sample.target,O:typeAs(sample.target),1)
   end
   self.input:resize(sample.input:size()):copy(sample.input) 
   target_rep = torch.repeatTensor(sample.target,self.opt.Nw,1):transpose(1,2)
   self.target:resize(target_rep:size()):copy(target_rep)
   return fake_examples
end

function Trainer:learningRate(epoch)
   -- Training schedule
   local decay = 0
   if self.opt.dataset == 'imagenet' then
      decay = math.floor((epoch - 1) / 30)
   elseif self.opt.dataset == 'BBCnet' then
      decay = math.floor((epoch - 1) / 5) -- I drop it by half every 5 epochs
   elseif self.opt.dataset == 'cifar10' then
      decay = epoch >= 122 and 2 or epoch >= 81 and 1 or 0
   elseif self.opt.dataset == 'cifar100' then
      decay = epoch >= 122 and 2 or epoch >= 81 and 1 or 0
   end
   return self.opt.LR * math.pow(0.5, decay)
end

return M.Trainer
