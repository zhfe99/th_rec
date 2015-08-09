local goo = require 'Models.goo'

local nC = 1000

local model = goo.new(nC, 2, true, 'xavier_caffe')

local loss = nn.ClassNLLCriterion()

local nEpo = 90

local nEpoSv = 30

local lrs = {
  {  1,   30, 1e-2, 5e-4},
  { 31,   60, 1e-3, 5e-4},
  { 61, nEpo, 1e-4, 5e-4}
}

local batchSiz = 100

local mom = 0.9

return model, loss, nEpo, nEpoSv, batchSiz, lrs, mom
