--
--  Copyright (c) 2016, Facebook, Inc.
--  All rights reserved.
--
--  This source code is licensed under the BSD-style license found in the
--  LICENSE file in the root directory of this source tree. An additional grant
--  of patent rights can be found in the PATENTS file in the same directory.
--
--  BBCNet dataset loader
--

local image = require 'image'
local paths = require 'paths'
local t = require 'datasets/transforms'
local ffi = require 'ffi'

local M = {}
local BBCnetDataset = torch.class('resnet.BBCnetDataset', M)

function BBCnetDataset:__init(imageInfo, opt, split)
   self.imageInfo = imageInfo[split]
   self.opt = opt
   self.split = split
   self.dir = opt.data
   print(self.imageInfo)
   assert(paths.dirp(self.dir), 'directory does not exist: ' .. self.dir)
end

function BBCnetDataset:get(i)
   -- At this stage, images should be 122x122. They will become 112x112 by random croppings. When testing this cropping is centered (non-random).  
   local path = ffi.string(self.imageInfo.imagePath[i]:data())
   --local Bb = self.imageInfo.imageBb[i] -- Use this to start from the images (you can use a fixed Bounding Box, since the images are already certered and scaled).
   --local images = self:_loadImage(paths.concat(self.dir, path),Bb) -- Use this to start from the images.
   local class = self.imageInfo.imageClass[i]

   local p1 = self.opt.imagecrop
   local p2 = paths.basename(string.sub(path,1,string.len(path)-8))
   local className = string.sub(p2,1,-7)
   local p3 = p1 .. className .. "/" .. self.split .. "/" .. p2 .. ".t7"
   
   --images = images*255 -- Use this to start from the images.
   --torch.save(p3,images:type('torch.ByteTensor')) -- Use this	to start from the images.

   --Assuming you have saved the images in as torch.ByteTensor :
   local images = torch.load(p3)
   images = (images:float())/255

   return {
      input = images,
      target = class,
   }
end

function BBCnetDataset:_loadImage(p, Bb)
   local Nfr = self.opt.Nfr
   local Dmarg = self.opt.Dmarg
   local Do = self.opt.Dimage + 2*Dmarg -- 112 + 2*5 = 122 
   local tocrop = torch.Tensor{80,116,176,212} -- or use Bb variable to define you our cropping coordinates
   
   local collection = torch.Tensor(Nfr,1,Do,Do)
   local i=0
   for i=1,Nfr do
       local path_i = string.sub(p,1,string.len(p)-7) .. string.format("%03d",i) .. ".png" -- GRAYSCALE ASSUMED
       local ok, input = pcall(function()
          return image.load(path_i, 1, 'float')
       end)
       
       input = image.crop(input,tocrop[1],tocrop[2],tocrop[3],tocrop[4])
       input = image.scale(input,Do,Do,'bilinear')
     
       collection[{{i},{},{},{}}] = input

   end
   return collection
end

function BBCnetDataset:size()
   return self.imageInfo.imageClass:size(1)
end

-- Computed from random subset of BBCNet training images
local meanstd = {
   mean = { 0.4161},
   std = { 0.1688 },
} -- These stats were estimated using a very small subset. Alternatively, you can use your own or simply mean = 0.5, std = 1.0

function BBCnetDataset:preprocess()
   if self.split == 'train' then
      return t.Compose{
         t.MyRandomCrop(self.opt.Dimage),
         t.ColorNormalize(meanstd),
         t.HorizontalFlip(0.5),
      }
   elseif self.split == 'val' or self.split == 'test' then
      return t.Compose{
         t.ColorNormalize(meanstd),
         t.CenterCrop(self.opt.Dimage),
      }
   else
      error('invalid split: ' .. self.split)
   end
end

return M.BBCnetDataset
