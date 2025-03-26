\section{API Module}\label{sec:API}

This module provides the web server functionality for the FruitLens application. We used the Scotty library to create a webserver that listens to the given port number

\hide{
\begin{code}
{-# LANGUAGE OverloadedStrings #-}

module API
  ( startServer,
    predictImage,
    MyImage (..),
  )
where

import Codec.Picture
import Codec.Picture.Extra (crop, scaleBilinear)
import Data.Aeson
import qualified Data.ByteString as BS
import qualified Data.ByteString.Base16 as B16
import qualified Data.ByteString.Base64 as B64
import qualified Data.ByteString.Char8 as BC
import Data.List.Split (chunksOf)
import Data.Vector.Storable (toList)
-- import NeuralNetwork (predictFruit)
import Web.Scotty (ActionM, get, html, json, jsonData, param, post, scotty)
\end{code}
}

First we define the \texttt{MyImage} type which is a newtype wrapper around a string. This is used to parse and serialize JSON objects with an \texttt{image} field.

\begin{code}
newtype MyImage = MyImage {value :: String} deriving (Show)

instance FromJSON MyImage where
  parseJSON = withObject "Image" $ \o -> MyImage <$> o .: "image"

instance ToJSON MyImage where
  toJSON (MyImage value) = object ["image" .= value]
\end{code}

Then \texttt{startServer} is defined, this function starts the web server on the specified port. The server listens for POST requests to \texttt{/api/fruitlens} and GET requests to the root path. The \texttt{predictImage} function is called when a POST request is made to \texttt{/api/fruitlens}.

\begin{code}
-- | Start the web server on the specified port
startServer :: Int -> IO ()
startServer port = scotty port $ do
  post "/api/fruitlens" predictImage
  get "/" $ html "<h1>FruitLens API</h1><p>Send POST requests to /api/fruitlens with base64 encoded images</p>"

\end{code}



\begin{code}

-- | Process an uploaded image
predictImage :: ActionM ()
predictImage = do
  (MyImage value) <- jsonData
  let decoded = B64.decode (BC.pack value)
  case decoded of
    Left err -> Web.Scotty.json $ object ["error" .= ("Invalid Base64: " ++ err)]
    Right byteArray -> do
      case decodeImage byteArray of
        Left err -> Web.Scotty.json $ object ["error" .= ("Image decoding failed: " ++ err)]
        Right dynImage -> do
          let normalizedPixels = convertImage dynImage

          -- Predict fruit type using our neural network
          -- let fruitType = predictFruit normalizedPixels

          -- Return both the image data and the prediction
          Web.Scotty.json $
            object
              [ "prediction" .= ("fruitType" :: String),
                "confidence" .= (0.85 :: Float), -- Placeholder confidence value
                "imageData"
                  .= object
                    [ "width" .= imageWidth converted,
                      "height" .= imageHeight converted
                    ]
              ]

\end{code}

The convertImage function takes a DynamicImage and converts it to a 100x100 3D list of Floats. Each element in the list is a pixel represented as a list of three floats, one for each RGB value, normalized to values between 0 and 1.

\begin{code}

convertImage :: DynamicImage -> [[[Float]]]
convertImage dynImage = 
  let converted = convertRGB8 dynImage
      square = toSquare converted --  Make it square before scaling to avoid distortion
      scaled = scaleBilinear 100 100 square -- Scale to 100 by 100 image
      rgbList = toList (imageData scaled) -- All RGB values in a row, so 30000 elements
    in map (\x - > x / 255) rgbList
\end{code}

Lastly, a small function \texttt{toSquare} is defined to crop an image to a centered square of the original image.

\begin{code}

toSquare :: Image PixelRGB8 -> Image PixelRGB8
toSquare img =
  let w = imageWidth img
      h = imageHeight img
      size = min w h
      x = (w - size) `div` 2
      y = (h - size) `div` 2
    in crop x y size size img

\end{code}
