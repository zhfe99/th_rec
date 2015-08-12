local alex = require 'Models.alex'

local nC = 1000

-- k20
local batchSiz = 128
if #opt.gpus > 1 then
  batchSiz = 256
end

-- k40
-- local batchSiz = 256
-- if #opt.gpus > 1 then
--   batchSiz = 512
-- end

local bufSiz = batchSiz * #opt.gpus

local model = alex.new(nC, opt.gpus, true, 'xavier_caffe')

local loss = nn.ClassNLLCriterion()

model:cuda()
loss:cuda()

local nEpo = 90

local nEpoSv = 40

local lrs = {
  {  1,   30, 1e-2, 5e-4},
  { 31,   60, 1e-3, 5e-4},
  { 61, nEpo, 1e-4, 5e-4}
}

local sampleSiz = {3, 224, 224}

-- init optimization state
local optStat = {
  learningRate = 0.01,
  momentum = 0.9,
  weightDecay = 5e-4,
  learningRateDecay = 0.0,
  dampening = 0.0
}

----------------------------------------------------------------------
-- Get the parameters for each epoch.
--
-- Input
--   epoch  -  epoch id
local function parEpo(epoch)
  for _, row in ipairs(lrs) do
    if epoch >= row[1] and epoch <= row[2] then
      return {learningRate=row[3], weightDecay=row[4]}, epoch == row[1]
    end
  end
end

return model, loss, nEpo, nEpoSv, batchSiz, bufSiz, sampleSiz, optStat, parEpo
