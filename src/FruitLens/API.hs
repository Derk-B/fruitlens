{-# LANGUAGE OverloadedStrings #-}

module FruitLens.API
  ( startServer
  , processImage
  , MyImage(..)
  ) where

import Web.Scotty (ActionM, scotty, post, jsonData, json, get, html, param)
import Data.Aeson 
import qualified Data.ByteString.Base64 as B64
import qualified Data.ByteString as BS 
import qualified Data.ByteString.Char8 as BC
import qualified Data.ByteString.Base16 as B16
import Codec.Picture
import Data.Vector.Storable (toList)
import Data.List.Split (chunksOf)

import FruitLens.NeuralNetwork (predictFruit)

-- | Image data type for JSON serialization/deserialization
newtype MyImage = MyImage { value :: String } deriving (Show)

instance FromJSON MyImage where
    parseJSON = withObject "Image" $ \o -> MyImage <$> o .: "image"

instance ToJSON MyImage where 
    toJSON (MyImage value) = object ["image" .= value]

-- | Start the web server on the specified port
startServer :: Int -> IO ()
startServer port = scotty port $ do
    post "/api/fruitlens" processImage
    get "/" $ html "<h1>FruitLens API</h1><p>Send POST requests to /api/fruitlens with base64 encoded images</p>"
        
-- | Process an uploaded image
processImage :: ActionM ()
processImage = do
    (MyImage value) <- jsonData  
    let decoded = B64.decode (BC.pack value)
    case decoded of
        Left err -> Web.Scotty.json $ object ["error" .= ("Invalid Base64: " ++ err)] 
        Right byteArray -> do 
          case decodeImage byteArray of
                Left err -> Web.Scotty.json $ object ["error" .= ("Image decoding failed: " ++ err)]
                Right dynImage -> do
                    let converted = convertRGB8 dynImage
                    let rgbList = toList (imageData converted)
                    let tupleList = chunksOf 3 rgbList
                    let tupleList2d = chunksOf (imageWidth converted) tupleList
                    
                    -- Convert pixel values from Word8 (0-255) to Float (0-1)
                    let normalizedPixels = map (map (\[r,g,b] -> [fromIntegral r / 255, fromIntegral g / 255, fromIntegral b / 255])) tupleList2d
                    
                    -- Predict fruit type using our neural network
                    let fruitType = predictFruit normalizedPixels
                    
                    -- Return both the image data and the prediction
                    Web.Scotty.json $ object [
                        "prediction" .= fruitType,
                        "confidence" .= (0.85 :: Float),  -- Placeholder confidence value
                        "imageData" .= object [
                            "width" .= imageWidth converted,
                            "height" .= imageHeight converted
                        ]
                      ] 