#!/usr/bin/env th
-- Generate lmdb.
--
-- History
--   create  -  Feng Zhou (zhfe99@gmail.com), 08-03-2015
--   modify  -  Feng Zhou (zhfe99@gmail.com), 08-10-2015

require 'eladtools'
require 'image'
require 'xlua'
require 'lmdb'
local ffi = require 'ffi'
local lib = require 'lua_lib'
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
cmd:option('-sizMa', 256, 'maximum image size')
local opt = cmd:parse(arg)

-- data
local dat = ThDat(opt.dbe, opt.ver)
local PATH = dat.PATH
cmd:log(PATH.logPath)

----------------------------------------------------------------------
-- Generate lmdb from a list of files.
--
-- Input
--   env       -  lmdb env
--   imgFold   -  image fold
--   imgList   -  image list
--   meanPath  -  mean path (optional)
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
    local imgNm = parts[1]
    local c = tonumber(parts[2]) + 1

    -- load image
    local imgPath = imgFold .. '/' .. imgNm
    local filename = ffi.string(imgPath)
    local img0 = lib.imgLoad(filename)

    -- rescale
    local img = lib.imgSizNew(img0, opt.sizMa)

    -- update mean & std
    for j = 1, 3 do
      me[j] = me[j] + img[j]:float():mean()
      std[j] = std[j] + img[j]:float():std()
    end

    -- compress
    if opt.cmp then
      img = image.compressJPG(img)
    end

    -- store
    local key = string.format('%07d', i)
    local data = {img = img, c = c, path = parts[1]}
    cursor:put(key, data, lmdb.C.MDB_NODUPDATA)
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
  print(string.format('mean: %.5f, %.5f, %.5f', me[1], me[2], me[3]))
  print(string.format('std : %.5f, %.5f, %.5f', std[1], std[2], std[3]))
  if meanPath then
    print('saving mean')
    torch.save(meanPath, meanInfo)
  end
end

-- create tr lmdb
assert(not paths.dirp(PATH.trLmdb), string.format('%s exists', PATH.trLmdb))
local trEnv = lmdb.env {
  Path = PATH.trLmdb,
  Name = 'train'
}
local trImgList = lib.loadLns(PATH.trListCaf)
genLmdbFromList(trEnv, PATH.dataFold .. '/train', trImgList, PATH.meanPath)

-- test
assert(not paths.dirp(PATH.teLmdb))
local teEnv = lmdb.env {
  Path = PATH.teLmdb,
  Name = 'test'
}
local teImgList = lib.loadLns(PATH.teListCaf)
genLmdbFromList(teEnv, PATH.dataFold .. '/test', teImgList)
