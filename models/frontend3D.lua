local nn = require 'nn'
local cunn = require 'cunn'
local cudnn = require 'cudnn'

local function createModel(opt)
   local model = nn.Sequential()
   model:add(cudnn.VolumetricConvolution(1, 64, 5, 7, 7, 1, 2, 2, 2, 3, 3))
   model:add(cudnn.VolumetricBatchNormalization(64))
   model:add(nn.ReLU(true))
   model:add(cudnn.VolumetricMaxPooling(1,3,3,1,2,2,0,1,1))
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
   ConvInitV('cudnn.VolumetricConvolution')
   for k,v in pairs(model:findModules('nn.Linear')) do
      v.bias:zero()
   end
   model:cuda()
   return model
end

return createModel