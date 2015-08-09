local alex = require 'Models.alex'

local nC = 1000

local model = alex.new(nC, 1, true, 'xavier_caffe')

local loss = nn.ClassNLLCriterion()

local nEpo = 90

local nEpoSv = 30

local lrs = {
  {  1,   30, 1e-2, 5e-4},
  { 31,   60, 1e-3, 5e-4},
  { 61, nEpo, 1e-4, 5e-4}
}

local batchSiz = 128

local mom = 0.9

return model, loss, nEpo, nEpoSv, batchSiz, lrs, mom
