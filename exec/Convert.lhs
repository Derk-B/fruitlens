\begin{code}

{-# HLINT ignore "Use first" #-}
{-# LANGUAGE TupleSections #-}
{-# OPTIONS_GHC -Wno-unrecognised-pragmas #-}

module Convert where

import FruitLens (trainModel, Image, newModelFC, newModelCNN, predictFruit, evaluateModel, loadModel, saveModel, convertImageForCNN)
import Codec.Picture
import qualified Data.Bifunctor
import qualified Data.Functor
import Data.List.Split (chunksOf)
import Data.Maybe (mapMaybe)
import Data.Vector.Storable (toList)
import GHC.Word (Word8)
import System.Directory

requiredDirectories :: [String]
requiredDirectories =
  [ "Apple 6/", -- green apple 500
    "Apple 9/", -- rotten apple 700
    "Apple 10/", -- red apple
    "Banana 1/",
    "Banana 3/"
    -- ,"Pear 1/" -- 500
  ]

directoriesWithLabels :: [(String, String)]
directoriesWithLabels = zip requiredDirectories ["apple", "apple", "apple", "banana", "banana"]

processImage :: IO (Either String DynamicImage) -> IO (Maybe [Word8])
processImage ioImg = do
  eitherImg <- ioImg
  case eitherImg of
    Left err -> do
      putStrLn $ "Error: " ++ err
      return Nothing
    Right img -> do
      let convertedImage = convertRGB8 img
      let imageAsRGBList = toList (imageData convertedImage)
      return $ Just imageAsRGBList

getImagesAndLabels :: String -> IO ([[Float]], [[Float]])
getImagesAndLabels path = do
  let dirPath = "fruits-360/" ++ path :: FilePath
  allFiles <- mapM (\(dp, l) -> listDirectory (dirPath ++ dp) >>= \fps -> return $ map (\f -> (dirPath ++ dp ++ f, l)) fps) directoriesWithLabels Data.Functor.<&> concat
  let allImages = map (\(fp, l) -> (readImage fp, l)) allFiles :: [(IO (Either String DynamicImage), String)]
  let processedImages = map (\(img, l) -> (processImage img, l)) allImages

  maybeImgsAndLabels <-
    mapM
      ( \(ioImg, label) -> do
          maybeBytes <- ioImg
          return (maybeBytes, label)
      )
      processedImages

  let word8ListToFloat = map (\b -> fromIntegral b / 255 :: Float)
  let labelStrToFloats l
        --   | l == "apple"    = [1.0, 0.0, 0.0]
        --   | l == "banana"   = [0.0, 1.0, 0.0]
        --   | otherwise       = [0.0, 0.0, 1.0]
        | l == "apple" = [1.0, 0.0]
        | otherwise = [0.0, 1.0]

  let validImgsAndLabels = mapMaybe (\(maybeImg, l) -> fmap (,l) maybeImg) maybeImgsAndLabels
  let imgAsFloastAndLabels = map (Data.Bifunctor.bimap word8ListToFloat labelStrToFloats) validImgsAndLabels
  let seperateBytesAndLabels = unzip imgAsFloastAndLabels

  return seperateBytesAndLabels



convert :: IO ()
convert = do
  (trainI, trainL) <- getImagesAndLabels "Training/"
  (testI, testL) <- getImagesAndLabels "Test/"

  let trainingData :: [(FruitLens.Image, [Float])]
      trainingData = zip (map convertImageForCNN trainI) trainL

  let testData :: [(FruitLens.Image, [Float])]
      testData = zip (map convertImageForCNN testI) testL

  -- Check if a saved model exists
  modelExists <- doesFileExist "trained_model.bin"
  _ <- if modelExists
    then do
      putStrLn "Loading saved model..."
      loadModel "trained_model.bin"
    else do
      initialModel <- newModelFC
      trainedModel <- trainModel initialModel trainingData 10 0.01
      evaluateModel trainedModel testData
      saveModel "trained_model.bin" trainedModel
      putStrLn "Model saved to model.bin."
      return trainedModel

  return ()

\end{code}