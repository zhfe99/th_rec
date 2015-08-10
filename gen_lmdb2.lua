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
cmd:text()
cmd:text('Generate lmdb for training and testing data')
cmd:text()
cmd:text('Options:')
cmd:addTime()
cmd:option('-dbe', 'car', 'database name')
cmd:option('-ver', 'v1c', 'version')
cmd:option('-cmp', true, 'compress or not')
cmd:option('-size', 256, 'image size')
local opt = cmd:parse(arg)

-- data
local dbe = opt.dbe
local ver = opt.ver
local dat = ThDat(dbe, ver)
local PATH = dat.PATH
local DATA = dat.DATA
cmd:log(PATH.logPath)

-- config
local confPath = string.format('Models/%s_%s_conf.lua', dbe, ver)
local config = paths.dofile(confPath)

----------------------------------------------------------------------
-- Pre-process image.
--
-- Input
--   img0  -  original image
--
-- Output
--   img   -  new img
local function PreProcess(img0)
  --minimum side of ImageSize
  local im = image.scale(img0, '^' .. config.ImageSize)

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
local function LoadImgData(filename)
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
--   env       -  lmdb env
--   imgFold   -  image fold
--   imgList   -  image list
--   meanPath  -  mean path
function genLmdbFromList(env, imgFold, imgList, meanPath)
  env:open()
  local txn = env:txn()
  local cursor = txn:cursor()
  local me = {0, 0, 0}
  local std = {0, 0, 0}

  -- each image
  local nImg = #imgList
  for i = 1, nImg do
    -- parse
    local ln = imgList[i]
    local parts = lib.split(ln, ' ')
    local imgPath = imgFold .. '/' .. parts[1]
    local c = tonumber(parts[2]) + 1
    local filename = ffi.string(imgPath)
    local img = LoadImgData(filename)

    -- update mean & std
    for j = 1, 3 do
      me[j] = me[j] + img[j]:float():mean()
      std[j] = std[j] + img[j]:float():std()
    end

    -- compress
    if opt.cmp then
      img = image.compressJPG(img:float() / 255)
    end

    -- store
    local data = {img = img, c = c, path = parts[1]}
    cursor:put(config.Key(i), data, lmdb.C.MDB_NODUPDATA)
    if i % 1000 == 0 then
      txn:commit()
      xlua.print(env:stat())
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
  local meanInfo = {
    me = me,
    std = std
  }
  print(string.format('mean: %.3f, %.3f, %.3f', me[1], me[2], me[3]))
  print(string.format('std : %.3f, %.3f, %.3f', std[1], std[2], std[3]))
  if meanPath then
    print('saving mean')
    torch.save(meanPath .. 'c', meanInfo)
  end
end

-- create tr lmdb
assert(not paths.dirp(PATH.trLmdb), string.format('%s exists', PATH.trLmdb))
local trEnv = lmdb.env {
  Path = PATH.trLmdb,
  Name = 'TrainDB'
}
local trImgList = lib.loadLns(PATH.trListCaf)
genLmdbFromList(trEnv, PATH.dataFold .. '/train', trImgList, PATH.meanPath)

-- test
assert(not paths.dirp(PATH.teLmdb))
local teEnv = lmdb.env {
  Path = PATH.teLmdb,
  Name = 'ValDB'
}
local teImgList = lib.loadLns(PATH.teListCaf)
genLmdbFromList(teEnv, PATH.dataFold .. '/test', teImgList)
