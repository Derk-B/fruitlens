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
import Codec.Picture.Extra (crop, scaleBilinear)
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
                Right dynImage -> do -- TODO make more compact? Is very clear now though
                    let converted = convertRGB8 dynImage
                    let square = toSquare converted --  Make it square before scaling to avoid distortion
                    let scaled = scaleBilinear 100 100 square -- Scale to 100 by 100 image
                    let rgbList = toList (imageData scaled) -- All RGB values in a row 
                    let tupleList = chunksOf 3 rgbList -- Convert to long list of [R, G, B] Tuples
                    let tupleList2d = chunksOf (imageWidth scaled) tupleList -- Convert to 2D list of [R, G, B] Tuples (top to bottom, left to right)
                    
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

toSquare :: Image PixelRGB8 -> Image PixelRGB8 -- Crop image to square, centered
toSquare img =
    let w = imageWidth img
        h = imageHeight img
        size = min w h
        x = (w - size) `div` 2
        y = (h - size) `div` 2
    in crop x y size size img