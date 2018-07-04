nn = require 'nn'
cunn = require 'cunn'
cudnn = require 'cudnn'
rnn = require 'rnn'
paths = require 'paths'

path_to_model = '/udisk/pszts/AV-ASR-data/BBC_Oxford/F_V_BBC_LSTM_resnet18_nopoolx2_f29_E2E/checkpoints/model_best.t7' --change this to the path your model is
-- You can also download a pretrained model from here: https://www.dropbox.com/s/85ppfyqtdih8fte/LipReadingNet_resnet18_f29.t7?dl=0 (18-layer instead of 34, works equally well).	      
path_to_vocab = 'vocab.txt' --you can do "wget http://www.robots.ox.ac.uk/~vgg/data/lip_reading/vocab.txt" to download it
dir_to_val_files = './' -- 5 .t7 files from val set are stored. Use image toolkit to see how they look. Check the code (datasets/BBCnet.lua) to see how to create them from the .mp4 files.

meanstd = {
   mean = { 0.4161},
   std = { 0.1688 },
}

M = torch.load(path_to_model):cuda()

file = io.open(path_to_vocab, "r")
vocab = {}
Nw = 0
for line in file:lines() do
    table.insert(vocab, line);
    Nw = Nw + 1
end
file:close()
print('vocab size: ' .. Nw)

val_files = {}
Nf = 0
for f in paths.files(dir_to_val_files) do
    if paths.extname(f) == 't7' then 
       table.insert(val_files,f)
       Nf = Nf + 1
    end
end

for i = 1,Nf do
    print('file name: ' .. val_files[i])
    P = paths.concat(dir_to_val_files,val_files[i])
    I = torch.load(P)
    I = I[{{},{6,117},{6,117}}] -- this is central cropping, you can also do it with image.crop (122,122) -> (112,112)
    I = (I:float())/255 -- they are stored as Byte tensors, turn them to float
    I = I:add(-meanstd.mean[1])
    I = I:div(meanstd.std[1])  
    I = I:resize(1,1,I:size(1),I:size(2),I:size(3)) -- first dim is the batch size.  I use batch_size=1 here to make this test easier to follow 
    O = M:forward(I:cuda())
    Om = O:mean(2):view(Nw) -- average over frame log-posteriors
    y, im = torch.max(Om,1) -- get max and argmax
    print("Identified as " .. vocab[im[1]]) 
end
