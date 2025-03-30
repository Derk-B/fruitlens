\section{API Module}\label{sec:API}

This section is about the API module, which we implemented to improve ease of use and make it possible to send requests from the webpage and the app. We used the Scotty library to create a webserver that listens to the given port number

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
import qualified Data.ByteString.Base64 as B64
import qualified Data.ByteString.Char8 as BC
import Data.Vector.Storable (toList)
import AI (loadModel, predictFruit, NeuralNetwork, FruitType(Apple, Banana))
import Convert (convertImageForCNN)
import Data.List (maximumBy)
import Data.Ord (comparing)
import Web.Scotty (ActionM, get, html, json, jsonData, post, scotty, middleware)
import Network.Wai.Middleware.Cors
import qualified Network.Wai as Wai

appCorsResourcePolicy :: CorsResourcePolicy
appCorsResourcePolicy = simpleCorsResourcePolicy
  { corsOrigins        = Just (["http://localhost:3000"], True) -- Allow only your website
  , corsMethods        = ["GET", "POST", "PUT", "DELETE"]      -- Allowed methods
  , corsRequestHeaders = ["Content-Type"]                      -- Allowed headers
  }
\end{code}
}

First we define the \texttt{MyImage} type which is a newtype wrapper around a string. This is used to parse and serialize JSON objects with an \texttt{image} field.

\begin{code}
newtype MyImage = MyImage {value :: String} deriving (Show)

instance FromJSON MyImage where
  parseJSON = withObject "Image" $ \o -> MyImage <$> o .: "image"

instance ToJSON MyImage where
  toJSON (MyImage v) = object ["image" .= v]
\end{code}

Then \texttt{startServer} is defined, this function starts the web server on the specified port. The server listens for POST requests to \texttt{/api/fruitlens} and GET requests to the root path. The \texttt{predictImage} function is called when a POST request is made to \texttt{/api/fruitlens}.

\begin{code}
-- | Start the web server on the specified port
startServer :: Int -> IO ()
startServer port = do
  model <- loadModel "trained_model.bin"
  scotty port $ do
    middleware $ cors (const $ Just appCorsResourcePolicy)
    post "/api/fruitlens" $ predictImage model
    get "/" $ html "<h1>FruitLens API</h1><p>Send POST requests to /api/fruitlens with base64 encoded images</p>"

\end{code}

The \texttt{predictImage} function takes the jsonData, packs it and tries to decode it from Base64 to ByteString. If an error occurs, the error is simply returned in JSON by the API. Then, the function tries to decode the image to a \texttt{dynamicImage} from the \texttt{JuicyPixels} library, if that throws an error, the API, again, simply returns an error to the request. If the image can be decoded, the image is converted to an array of floats, which will be sent to the neural network to predict the fruit with. Lastly, the API returns the prediction to the requester. 

\begin{code}
-- | Predict the fruit type from an image using the neural network
predictImage :: NeuralNetwork -> ActionM ()
predictImage model = do
  (MyImage v) <- jsonData
  let decoded = B64.decode (BC.pack v)
  case decoded of
    Left err -> Web.Scotty.json $ object ["error" .= ("Invalid Base64: " ++ err)]
    Right byteArray -> do
      case decodeImage byteArray of
        Left err -> Web.Scotty.json $ object ["error" .= ("Image decoding failed: " ++ err)]
        Right dynImage -> do
          let normalizedPixels = convertImage dynImage
              cnnFormatPixels = convertImageForCNN normalizedPixels
              prediction = predictFruit model cnnFormatPixels
              fruitType = case prediction of
                Apple -> "apple"
                Banana -> "banana"

          -- Return the prediction (confidence is now handled internally by the AI module)
          Web.Scotty.json $
            object
              [ "prediction" .= (fruitType :: String),
                "confidence" .= (1.0 :: Float)  -- Placeholder
              ]
\end{code}

The \texttt{convertImage} function takes a DynamicImage and converts it to a list of 30.000 Floats (100x100 pixels, 3 RGB values per pixel). Each element in the list is a pixel represented as a list of three floats, one for each RGB value, normalized to values between 0 and 1. The convertRGB8 and scaleBilinear functions are from the \texttt{JuicyPixels-extra} library.

\begin{code}
convertImage :: DynamicImage -> [Float]
convertImage dynImage = 
  let converted = convertRGB8 dynImage
      square = toSquare converted --  Make it square before scaling to avoid distortion
      scaled = scaleBilinear 100 100 square -- Scale to 100 by 100 image
      rgbList = toList (imageData scaled) -- All RGB values in a row, so 30000 elements
    in map (\x -> fromIntegral x / 255) rgbList
\end{code}

Lastly, a small function \texttt{toSquare} is defined to crop an image to a centered square of the original image. The crop funtion is from the  \texttt{JuicyPixels-extra} library.

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
