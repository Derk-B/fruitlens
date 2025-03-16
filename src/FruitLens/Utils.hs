module FruitLens.Utils
  ( gauss
  , readMNISTLabels
  , readMNISTImages
  ) where

import Codec.Compression.GZip (decompress)
import qualified Data.ByteString.Lazy as BL
import System.Random

-- | Generate a random number from a Gaussian distribution
gauss :: Float -> IO Float
gauss scale = do
  x1 <- randomIO
  x2 <- randomIO
  return $ scale * sqrt (- (2 * log x1)) * cos (2 * pi * x2)

-- | Read MNIST labels from a file
-- This is a placeholder function that will be implemented later
readMNISTLabels :: FilePath -> IO [Int]
readMNISTLabels _ = return [0, 1, 2, 3, 4, 5, 6, 7, 8, 9] -- Placeholder implementation

-- | Read MNIST images from a file
-- This is a placeholder function that will be implemented later
readMNISTImages :: FilePath -> IO [[Float]]
readMNISTImages _ = return (replicate 10 (replicate 784 0)) -- Placeholder implementation 