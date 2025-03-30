\section{Utilities Module}\label{sec:Utils}

This module provides utility functions for the FruitLens application.

\begin{code}
module Utils
  ( gauss,
    gaussian,
    readMNISTLabels,
    readMNISTImages,
  )
where

import Codec.Compression.GZip (decompress)
import qualified Data.ByteString.Lazy as BL
import System.Random

-- | Generate a random number from a Gaussian distribution
gauss :: Float -> IO Float
gauss scale = do
  x1 <- randomIO
  x2 <- randomIO
  return $ scale * sqrt (-(2 * log x1)) * cos (2 * pi * x2)

-- Gaussian function: G(x, y) = 1/(2*pi*sigma^2) * exp ( - (x^2+y^2)/(2*sigma^2) )
gaussian :: Float -> Float -> Float -> Float
gaussian x y sigma = 1 / (2 * pi * sigma * sigma) * exp (-((x*x + y*y) / (2 * sigma * sigma)))

-- | Read MNIST labels from a file
-- This is a placeholder function that will be implemented later
readMNISTLabels :: FilePath -> IO [Int]
readMNISTLabels _ = return [0, 1, 2, 3, 4, 5, 6, 7, 8, 9] -- Placeholder implementation

-- | Read MNIST images from a file
-- This is a placeholder function that will be implemented later
readMNISTImages :: FilePath -> IO [[Float]]
readMNISTImages _ = return (replicate 10 (replicate 784 0)) -- Placeholder implementation

\end{code}
