paths.dofile('models/alex.lua')

local model = newModel(196, 1)

local nEpo = 60

local lrs = {
  { 1,   18, 1e-2, 5e-4},
  {19,   29, 5e-3, 5e-4},
  {30,   43, 1e-3, 0},
  {44,   52, 5e-4, 0},
  {53, nEpo, 1e-4, 0},
}

return model, nEpo, lrs
