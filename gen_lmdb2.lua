#!/usr/bin/env th
-- Generate lmdb.
--
-- History
--   create  -  Feng Zhou (zhfe99@gmail.com), 08-03-2015
--   modify  -  Feng Zhou (zhfe99@gmail.com), 08-09-2015

require 'eladtools'
require 'image'
require 'xlua'
require 'lmdb'
local lib = require('lua_lib')
local gm = require 'graphicsmagick'
local ffi = require 'ffi'
local th = require 'lua_th'

-- argument
cmd = torch.CmdLine()
cmd:option('-dbe', 'car', 'database name')
cmd:option('-ver', 'v1c', 'version')
local params = cmd:parse(arg)

-- data
local dbe = params.dbe
local ver = params.ver
local dat = ThDat(dbe, ver)
local PATH = dat.PATH
local DATA = dat.DATA

-- config
local confPath = string.format('Models/%s_%s_conf.lua', dbe, ver)
local config = paths.dofile(confPath)

----------------------------------------------------------------------
-- Pre-process image.
--
-- Input
--   Img  -  original image
--
-- Output
--   img  -  new img
local PreProcess = function(Img)
  --minimum side of ImageSize
  local im = image.scale(Img, '^' .. config.ImageSize)

  if im:dim() == 2 then
    im = im:reshape(1, im:size(1), im:size(2))
  end
  if im:size(1) == 1 then
    im = torch.repeatTensor(im, 3, 1, 1)
  end
  if im:size(1) > 3 then
    im = im[{{1, 3}, {}, {}}]
  end
  return im
end

----------------------------------------------------------------------
-- Load image data.
--
-- Input
--   filename  -  file path
--
-- Output
--   img       -  image
local LoadImgData = function(filename)
  local ok, img = pcall(gm.Image, filename)

  -- error
  if not ok or img == nil then
    print('Image is buggy')
    print(filename)
    os.exit()
  end

  -- pre-process
  img = img:toTensor('byte', 'RGB', 'DHW')
  img = PreProcess(img)
  return img
end

----------------------------------------------------------------------
-- Generate lmdb from a list of files.
--
-- Input
--   env         -  lmdb env
--   charTensor  -  file name list
function LMDBFromList(env, foldPath, imgList, meanPath)
  env:open()
  local txn = env:txn()
  local cursor = txn:cursor()
  local me = {0, 0, 0}
  local std = {0, 0, 0}
  local nImg = #imgList
  for i = 1, nImg do
    -- parse
    local ln = imgList[i]
    local parts = lib.split(ln, ' ')
    local imgPath = foldPath .. '/' .. parts[1]
    local c = tonumber(parts[2]) + 1
    local filename = ffi.string(imgPath)
    local img = LoadImgData(filename)

    -- update mean & std
    for j = 1, 3 do
      me[j] = me[j] + img[j]:float():mean()
      std[j] = std[j] + img[j]:float():std()
    end

    -- store
    local data = {img = img, c = c, path = imgPath}
    cursor:put(config.Key(i), data, lmdb.C.MDB_NODUPDATA)
    if i % 1000 == 0 then
      txn:commit()
      print(env:stat())
      collectgarbage()
      txn = env:txn()
      cursor = txn:cursor()
    end
    xlua.progress(i, nImg)
  end
  txn:commit()
  env:close()

  -- update mean & std
  for j = 1, 3 do
    me[j] = me[j] / nImg
    std[j] = std[j] / nImg
  end

  -- save mean
  meanInfo = {
    me = me,
    std = std
  }
  if meanPath then
    torch.save(meanPath, meanInfo)
  end
  print(string.format('mean: %.3f, %.3f, %.3f', me[1], me[2], me[3]))
  print(string.format('std : %.3f, %.3f, %.3f', std[1], std[2], std[3]))
end

-- create tr lmdb
local trEnv = lmdb.env {
  Path = PATH.trLmdb,
  Name = 'TrainDB'
}
LMDBFromList(ValDB, PATH.dataFold .. '/test', lns)

local teEnv = lmdb.env {
  Path = PATH.teLmdb,
  Name = 'ValDB'
}

-- test
local lns = lib.loadLns(PATH.teListCaf)
LMDBFromList(ValDB, PATH.dataFold .. '/test', lns)


-- LMDBFromFilenames(TrainingFiles.Data, TrainDB)
