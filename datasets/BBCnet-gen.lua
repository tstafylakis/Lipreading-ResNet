--
--  Copyright (c) 2016, Facebook, Inc.
--  All rights reserved.
--
--  This source code is licensed under the BSD-style license found in the
--  LICENSE file in the root directory of this source tree. An additional grant
--  of patent rights can be found in the PATENTS file in the same directory.
--
--  Script to compute list of ImageNet filenames and classes
--
--  This generates a file gen/imagenet.t7 which contains the list of all
--  ImageNet training and validation images and their classes. This script also
--  works for other datasets arragned with the same layout.
--

local sys = require 'sys'
local ffi = require 'ffi'
local hdf5 = require 'hdf5'
local faceAlignment = require 'faceAlignment'
local M = {}

local function findClasses(dir)
   local dirs = paths.dir(dir)
   table.sort(dirs)
   table.remove(dirs,1)
   table.remove(dirs,1)
   --local dirs = {}
   --dirs[1] = "JAMES"; dirs[2] = "JUDGE"; dirs[3] = "JUSTICE";
   local classList = {}
   local classToIdx = {}
   for _ ,class in ipairs(dirs) do
      if not classToIdx[class] and class ~= '.' and class ~= '..' then
         table.insert(classList, class)
         classToIdx[class] = #classList
      end
   end

   -- assert(#classList == 1000, 'expected 1000 ImageNet classes')
   return classList, classToIdx
end

local function get_invTrans()
    local bdBox = torch.Tensor(4,1)
    bdBox[1] = 1; bdBox[2] = 1; bdBox[3] = 251; bdBox[4] = 251;
    local c = torch.cat(bdBox[1]+(bdBox[3]-bdBox[1])/2.0, bdBox[2]+(bdBox[4]-bdBox[2])/2.0)
    local s = torch.max(torch.cat(bdBox[3]-bdBox[1],bdBox[4]-bdBox[2]))/300
    local t = getTransform(c, s, 0, 256)
    return torch.inverse(t)
end

local function get_Bb(lm, invTrans)
   local M = torch.median(lm,1):view(lm:size(2), lm:size(3))
   local M = torch.cat(M[{{29},{}}],M[{{9},{}}],1)
   local M = (torch.cat(M,torch.ones(2,1):short(),2)):transpose(1,2):float()
   return (invTrans*M):sub(1,2):add(1e-4):transpose(1,2):round():contiguous():view(1,4)
end

local function findImages(dir, lmDir, split, classToIdx)
   local imagePath = torch.CharTensor()
   local imageClass = torch.LongTensor()

   ----------------------------------------------------------------------
   -- Options for the GNU and BSD find command
   -- local extensionList = {'jpg', 'png', 'jpeg', 'JPG', 'PNG', 'JPEG', 'ppm', 'PPM', 'bmp', 'BMP'}
   -- local extensionList = {'001.png'}
   -- local findOptions = ' -iname "*.' .. extensionList[1] .. '"'
   --for i=2,#extensionList do
   --   findOptions = findOptions .. ' -o -iname "*.' .. extensionList[i] .. '"'
   --end

   -- Find all the images using the find command
   
   local maxLength = -1
   local imagePaths = {}
   local imageClasses = {}
   local imageBb = {}
   local dirs = paths.dir(dir)
   table.sort(dirs)
   table.remove(dirs,1)
   table.remove(dirs,1)
   --local dirs = {}
   --dirs[1] = "JAMES"; dirs[2] = "JUDGE"; dirs[3] = "JUSTICE";
   
   
   local extensionList = '001.png'
   local findOptions = ' -iname "*' .. extensionList .. '"'
   local invTrans = get_invTrans()
   
   for _ ,className in ipairs(dirs) do
      local fn = 'find -L ' .. dir .. className .. '/' .. split .. findOptions
      local f = io.popen(fn)
      local lm_path = lmDir .. className .. '_' .. split .. '.h5'
      --print(fn)
      --print(lm_path)
      local hclass = hdf5.open(lm_path, 'r')
      local lm_tbl = hclass:read():all()
      hclass:close()
      -- Generate a list of all the images and their class
      while true do
         local line = f:read('*line')
         if not line then break end
         --print(line)
         
         -- local className = paths.basename(paths.dirname(line))
         local filename = paths.basename(line)
         local path = className .. '/' .. split .. '/' .. filename

         local classId = classToIdx[className]
         assert(classId, 'class not found: ' .. className)
         
         lm_name = string.sub(filename,1,string.len(filename)-8)
         --print(lm_name)
         lm = lm_tbl[lm_name]
         Bb = get_Bb(lm, invTrans)
         table.insert(imagePaths, path)
         table.insert(imageClasses, classId)
         table.insert(imageBb, Bb)
         maxLength = math.max(maxLength, #path + 1)
      end
      
      f:close()
   end
   -- Convert the generated list to a tensor for faster loading
   local nImages = #imagePaths
   local imagePath = torch.CharTensor(nImages, maxLength):zero()
   for i, path in ipairs(imagePaths) do
      ffi.copy(imagePath[i]:data(), path)
   end
   
   local imageBbm = torch.IntTensor(nImages, 4)
   for i, path in ipairs(imageBb) do
      imageBbm[{{i},{}}] = imageBb[i]
   end
   
   local imageClass = torch.LongTensor(imageClasses)
   return imagePath, imageClass, imageBbm
end

function M.exec(opt, cacheFile)
   -- find the image path names
   local imagePath = torch.CharTensor()  -- path to each image in dataset
   local imageClass = torch.LongTensor() -- class index of each image (class index in self.classes)
   local imageBb = torch.IntTensor()
   
   local trainDir = opt.data
   local valDir = opt.data   
   local lmDir = opt.lmDir
   
   assert(paths.dirp(trainDir), 'train directory not found: ' .. trainDir)
   assert(paths.dirp(valDir), 'val directory not found: ' .. valDir)
   assert(paths.dirp(lmDir), 'landmark directory not found: ' .. lmDir)
   
   print("=> Generating list of images")
   local classList, classToIdx = findClasses(trainDir)

   print(" | finding all validation images")
   local valImagePath, valImageClass, valImageBb = findImages(valDir, lmDir, 'val', classToIdx)

   print(" | finding all training images")
   local trainImagePath, trainImageClass, trainImageBb = findImages(trainDir, lmDir, 'train', classToIdx)

   local info = {
      basedir = opt.data,
      lmDir = opt.lmDir,
      classList = classList,
      train = {
         imagePath = trainImagePath,
         imageClass = trainImageClass,
         imageBb = trainImageBb,
      },
      val = {
         imagePath = valImagePath,
         imageClass = valImageClass,
         imageBb = valImageBb,
      },
   }

   print(" | saving list of images to " .. cacheFile)
   torch.save(cacheFile, info)
   return info
end

return M
