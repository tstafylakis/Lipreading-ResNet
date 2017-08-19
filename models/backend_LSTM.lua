local nn = require 'nn'
require 'cunn'
require 'rnn'

local function createModel(opt)
  local function create_brnn(inputDim, hiddenDim, batchFirst)
    forwardModule1 = nn.SeqLSTM(inputDim, hiddenDim)
    backwardModule1 = nn.SeqLSTM(inputDim, hiddenDim)
    forwardModule2 = nn.SeqLSTM(hiddenDim, hiddenDim)
    backwardModule2 = nn.SeqLSTM(hiddenDim, hiddenDim)
    
    dim = 1
    
    local forward = nn.Sequential()
    forward:add(forwardModule1)
    forward:add(forwardModule2)
    local backward = nn.Sequential()
    backward:add(nn.SeqReverseSequence(dim)) -- reverse
    backward:add(backwardModule1)
    backward:add(backwardModule2)
    backward:add(nn.SeqReverseSequence(dim))

    local concat = nn.ConcatTable()
    concat:add(forward):add(backward)

    local brnn = nn.Sequential()
    brnn:add(concat)
    brnn:add(nn.JoinTable(-1))

    if(batchFirst) then
        -- Insert transposes before and after the brnn.
        brnn:insert(nn.Transpose({1, 2}), 1)
        brnn:insert(nn.Transpose({1, 2}))
    end
    return brnn
  end
   
   local model = nn.Sequential()
   model:add(create_brnn(opt.inputDim, opt.hiddenDim, true))
   model:add(nn.Bottle(nn.Linear(opt.hiddenDim*2, opt.nClasses)))
   model:add(nn.Bottle(nn.LogSoftMax()))
   for k,v in pairs(model:findModules('nn.Linear')) do
      v.bias:zero()
   end
   model:cuda()
   return model
end

return createModel
