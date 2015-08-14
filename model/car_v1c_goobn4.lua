local goo = require 'Models.goo'

local nC = 196

local model = goo.new(nC, 1, true, 'xavier_caffe')

local loss = nn.ClassNLLCriterion()

local nEpo = 120

local nEpoSv = 40

local lrs = {
  {  1,   40, 1e-2, 5e-4},
  { 41,   80, 1e-3, 5e-4},
  { 81, nEpo, 1e-4, 5e-4}
}

local batchSiz = 50

local mom = 0.9

return model, loss, nEpo, nEpoSv, batchSiz, lrs, mom
