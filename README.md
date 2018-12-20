# [Banana For Scale](http://www.bananaforscale.lol)

by [Julia Chapman](https://www.linkedin.com/in/julia-chapman)

# Overview

In 2005, an internet meme was started when a woman, trying to sell a TV, put a banana in her sale ad as a unit of measurement. Since then, ['Banana For Scale'](https://knowyourmeme.com/memes/banana-for-scale) has grown in popularity and has been dubbed the ['yardstick of the internet.'](https://www.dailydot.com/unclick/banana-for-scale-meme-history/)

![meme_origin1](https://i.kym-cdn.com/photos/images/newsfeed/000/746/898/b48.jpg)

"Banana for Scale" has also been adapted to other objects. In this case, a double mattress!

![meme_origin2](https://i.kym-cdn.com/photos/images/newsfeed/001/235/568/cb1.jpg)

So this got me thinking, what if you could use a banana as an actual unit for scale in an image?

# Data

Using my [website](http://www.bananaforscale.lol), I asked friends and family to help my out by uploading images of themselves holding bananas and their height.

As of December 20th, I have collected 148 images to use in my dataset. The average Height was 5'7".

Due to human nature, I had some duplicate photos, blurry photos, and jokesters who posed two people with one banana.

# Model

Using my transfer learning model from [Capstone 2](https://github.com/jchapman3773/Capstone-2) (altered for regression) performed better than expected, but did not perform very well. In the training, the model quickly overfit after less than 10 epochs. The chosen best model, based on validation loss, had a RMSE of 6.38inches.

![model rmse](https://github.com/jchapman3773/Capstone-3/blob/master/graphics/Transfer_CNN_reg_rmse_hist.png)

Next, I moved onto object detection models.

Using the ImageAI python library, I was able to use a pretrained RetinaNet model to detect both people and bananas in my images. The RetinaNet model was pretrained on the COCO dataset from Microsoft. During this process, I lost about 10% of the data due to the model not being able to detect a banana in some of the images, even with a class probability threshold of 10%.

I was then able to take the output (bounding box corrdinates and class probabilities) and feed the information into a ElasticNetCV linear model, GradientBoostingRegressor, and RandomForestRegressor from sklearn. Many of the features had to be removed in the linear model due to collinearity in the data. 

The random forest model performed best:
```
R^2: 0.075
RMSE: 7.943
MAE: 4.5
```
The difference between the RMSE and MAE shows that there are some large error outliers in the predictions.

You can test my models for yourself on my website: [bananaforscale.lol](http://bananaforscale.lol)

# Future Work

As I gather more data, I would like to keep retraining my models and see if they improve.

On a practical scale, this model and the premise behind it could be used in retail for estimating a person's clothing size to help them with shopping at different brands. In that case, the 'ruler' should be something common, but with more standard size. Perhaps a scaled image on a wall of a dressing room, an object near the clothing racks, or a standardized household object to aid online clothing retail.
