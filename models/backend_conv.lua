local nn = require 'nn'
require 'cunn'

local Convolution = cudnn.SpatialConvolution
local TemporalConvolution = cudnn.TemporalConvolution
local Avg = cudnn.SpatialAveragePooling
local ReLU = nn.ReLU
local TMax = nn.TemporalMaxPooling
local TAvg = nn.TemporalAveragePooling
local Max = nn.SpatialMaxPooling
local SBatchNorm = nn.SpatialBatchNormalization
local BatchNorm = nn.BatchNormalization

local function createModel(opt)
   local bn_size = opt.inputDim
   model = nn.Sequential()
   model:add(TemporalConvolution(bn_size,2*bn_size,5,2,0))
   model:add(nn.Bottle(BatchNorm(2*bn_size)))
   model:add(ReLU(true))
   model:add(TMax(2,2))
   model:add(TemporalConvolution(2*bn_size,4*bn_size,5,2,0))
   model:add(nn.Bottle(BatchNorm(4*bn_size)))
   model:add(ReLU(true))
   model:add(nn.Mean(2))
   model:add(nn.Linear(4*bn_size, bn_size))
   model:add(BatchNorm(bn_size))
   model:add(ReLU(true))
   model:add(nn.Linear(bn_size, opt.nClasses))

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

   ConvInit('cudnn.SpatialConvolution')
   ConvInit('cudnn.TemporalConvolution')
   ConvInit('nn.SpatialConvolution')
   BNInit('fbnn.SpatialBatchNormalization')
   BNInit('cudnn.SpatialBatchNormalization')
   BNInit('nn.SpatialBatchNormalization')
   BNInit('cudnn.VolumetricBatchNormalization')
   BNInit('nn.BatchNormalization')

   for k,v in pairs(model:findModules('nn.Linear')) do
      v.bias:zero()
   end

   model:cuda()

   return model
end
return createModel