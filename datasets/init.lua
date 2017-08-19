--
--  Copyright (c) 2016, Facebook, Inc.
--  All rights reserved.
--
--  This source code is licensed under the BSD-style license found in the
--  LICENSE file in the root directory of this source tree. An additional grant
--  of patent rights can be found in the PATENTS file in the same directory.
--
--  BBCNet, ImageNet and CIFAR-10 datasets
--

local M = {}

local function isvalid(opt, cachePath)
   --local imageInfo = torch.load(cachePath)
   --if imageInfo.basedir and imageInfo.basedir ~= opt.data then
   --   return false
   --end
   return true
end

function M.create(opt, split)
   local ext1 = '.t7'
   local ext2 = '-gen.lua'
   if split == 'test' then
      ext1 = '-test.t7'
      ext2 = '-gen-test.lua'
   end
   local cachePath = paths.concat(opt.gen, opt.dataset .. ext1)  ---%%%%SOS%%%%
   print(cachePath)
   if not paths.filep(cachePath) or not isvalid(opt, cachePath) then
      --paths.mkdir('gen')
      print('Recreating ' .. split)
      local script = paths.dofile(opt.dataset .. ext2) ---%%%%SOS%%%%
      script.exec(opt, cachePath)
   end
   local imageInfo = torch.load(cachePath)
   --print(imageInfo)
   local Dataset = require('datasets/' .. opt.dataset)
   return Dataset(imageInfo, opt, split)
end

return M
