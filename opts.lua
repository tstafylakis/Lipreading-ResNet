--
--  Copyright (c) 2016, Facebook, Inc.
--  All rights reserved.
--
--  This source code is licensed under the BSD-style license found in the
--  LICENSE file in the root directory of this source tree. An additional grant
--  of patent rights can be found in the PATENTS file in the same directory.
--
local M = { }

function M.parse(arg)
   local p_prefix = ''
   local cmd = torch.CmdLine()
   cmd:text()
   cmd:text('Torch-7 ResNet Training script')
   cmd:text('See https://github.com/facebook/fb.resnet.torch/blob/master/TRAINING.md for examples')
   cmd:text()
   cmd:text('Options:')
    ------------ General options --------------------
   cmd:option('-data',       '/udisk/pszts/AV-ASR-data/BBC_Oxford/frames/',         'Path to dataset')
   cmd:option('-imagecrop',       '/udisk/pszts-ssd/AV-ASR-data/BBC_Oxford/imagecrop_1D/',         'Path to local images hdf5')
   cmd:option('-Nfr',       29,         'Number of frames per collection')
   cmd:option('-Nw',       29,         'Number of frames actually used')
   cmd:option('-Dimage',       112,         'Size of input images to CNN')
   cmd:option('-Dmarg',       5,         'Used for cropping images from 122 to 112')
   cmd:option('-dataset',    'BBCnet', 'Options: BBCnet | imagenet | cifar10 | cifar100')
   cmd:option('-manualSeed', 1,          'Manually set RNG seed')
   cmd:option('-nGPU',       1,          'Number of GPUs to use by default')
   cmd:option('-backend',    'cudnn',    'Options: cudnn | cunn')
   cmd:option('-cudnn',      'fastest',  'Options: fastest | default | deterministic')
   cmd:option('-gen',        'gen',      'Path to save generated files')
   ------------- Data options ------------------------
   cmd:option('-nThreads',        5, 'number of data loading threads')
   ------------- Training options --------------------
   cmd:option('-nEpochs',         50,       'Number of total epochs to run')
   cmd:option('-epochNumber',     1,       'Manual epoch number (useful on restarts)')
   cmd:option('-batchSize',       18,      'mini-batch size (1 = pure stochastic)')
   cmd:option('-testOnly',        'false', 'Run on validation set only')
   cmd:option('-tenCrop',         'false', 'Ten-crop testing')
   ------------- Checkpointing options ---------------
   cmd:option('-save',           'checkpoints', 'Directory in which to save checkpoints')
   cmd:option('-resume',         'checkpoints', 'Resume from the latest checkpoint in this directory')
   ---------- Optimization options ----------------------
   cmd:option('-LR',              0.003,   'initial learning rate')
   cmd:option('-momentum',        0.9,   'momentum')
   cmd:option('-weightDecay',     1e-4,  'weight decay')
   ---------- Model options ----------------------------------
   cmd:option('-netType',      'temp_conv', 'Defines the backend: temp_conv | LSTM_init | LSTM')
   cmd:option('-inputDim',      256)
   cmd:option('-hiddenDim',      256) -- for LSTM
   cmd:option('-depth',        34,       'ResNet depth: 18 | 34 | 50 | 101 | ...', 'number') -- Try 18 as well, equivalent results.
   cmd:option('-shortcutType', '',       'Options: A | B | C')
   cmd:option('-retrain',      'none',   'Path to model to retrain with, use it either to train the LSTM backend alone or the whole net with LSTM backend end-to-end. Go to models/init.lua to see how it is implemented.')
   cmd:option('-optimState',   'none',   'Path to an optimState to reload from')
   ---------- Model options ----------------------------------
   cmd:option('-shareGradInput',  'false', 'Share gradInput tensors to reduce memory usage') -- set to false
   cmd:option('-optnet',          'false', 'Use optnet to reduce memory usage')  --  set to false
   cmd:option('-resetClassifier', 'false', 'Reset the fully connected layer for fine-tuning') -- To use it, small changes of code are required
   cmd:option('-nClasses',         500,      'Number of classes in the dataset')
   cmd:text()
   
   local opt = cmd:parse(arg or {})

   opt.save = opt.save .. '_' .. opt.netType -- use different checkpoints dirs for each experiment
   opt.resume = opt.save

   opt.testOnly = opt.testOnly ~= 'false'
   opt.tenCrop = opt.tenCrop ~= 'false'
   opt.shareGradInput = opt.shareGradInput ~= 'false'
   opt.optnet = opt.optnet ~= 'false'
   opt.resetClassifier = opt.resetClassifier ~= 'false'

   if not paths.dirp(opt.save) and not paths.mkdir(opt.save) then
      cmd:error('error: unable to create checkpoint directory: ' .. opt.save .. '\n')
   end

   if opt.dataset == 'BBCnet' then
      -- Handle the most common case of missing -data flag
      local trainDir = opt.data
      if not paths.dirp(opt.data) then
         cmd:error('error: missing BBCNet data directory')
      elseif not paths.dirp(trainDir) then
         cmd:error('error: BBCNet missing `train` directory: ' .. trainDir)
      end
      -- Default shortcutType=B and nEpochs=90
      opt.shortcutType = opt.shortcutType == '' and 'B' or opt.shortcutType
      opt.nEpochs = opt.nEpochs == 0 and 90 or opt.nEpochs
   else
      cmd:error('unknown dataset: ' .. opt.dataset)
   end

   if opt.resetClassifier then
      if opt.nClasses == 0 then
         cmd:error('-nClasses required when resetClassifier is set')
      end
   end

   if opt.shareGradInput and opt.optnet then
      cmd:error('error: cannot use both -shareGradInput and -optnet')
   end

   return opt
end

return M
