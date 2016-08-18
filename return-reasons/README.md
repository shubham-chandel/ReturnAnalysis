# Return Analysis

## Usage

```python
>>> from analysis import ReturnAnalysis
```

### Load Data

Load data from specified `csv` or `pickle` format.

```python
>>> obj = ReturnAnalysis('Handsets')

>>> obj.loadData(file='returns.csv')
>>> obj.data_frame

   index      product_title     return_date_time  return_comments  \
0      2  Panasonic Eluga Z  2016-06-09 13:08:25  camera not good   

       return_reason return_product_category_name return_sub_reason  
0  DEFECTIVE_PRODUCT                     Handsets      CAMERA_ISSUE  
...
```

#### Parameters

- `Return Category`: Return category to be analyzed.

### Generate Return Reasons

Generate return reasons from (unsupervised) learning of users return comments.

```python
>>> obj.generateKeywords(minChar=3, maxWords=3, minFrequency=15)
>>> obj.keywords

0            received empty box
1           poor battery backup
2           battery backup poor
3            back cover missing
4           camera quality poor
5            head phone missing
6    network connectivity issue
7              different colour
...
```

#### Parameters

- `minChar`: Minimum number of characters required for keyword to be valid
- `maxWords`: Maximum number of words possible in a keyword
- `minFrequency`: Minimum number of times keyword should appear

### Best set of Return Reasons 

Cluster keywords to form best set of return reasons.

```python
>>> obj.clusterKeywords(algorithm='doc2vec')
>>> obj.clusters

0    [received empty box, nothing present box, empt...
1    [battery drain fast, battery drained, battery ...
2            [received different item, different item]
3    [poor battery backup, battery backup poor, poo...
4    [camera quality poor, poor camera quality, bad...
...

```

#### Parameters

- `algorithm`
	- Algorithm used to convert text to vector for analysis
	- Value: 'doc2vec' or 'wmd'


### Incorrectly Annotated Return-comment

Identify return comments which were Incorectly Annotated and predict correct category.

```python
>>> obj.incorrectAnnoated()
>>> obj.incorectdf

                                 return_comments          return_sub_reason             return_predicted_reason
8                                   touch screen  EXTERNAL_KEYS_NOT_WORKING                   TOUCH_NOT_WORKING
36                        video quality very bad          OVERHEATING_ISSUE                        CAMERA_ISSUE
55                                camera clarity          DUPLICATE_PRODUCT                        CAMERA_ISSUE
233 		mobile hanging also and getting m...             OS_MALFUNCTION                       HANGING_ISSUE
256 front camera not working and headset missing               CAMERA_ISSUE  MISSING_ACCESSORY and CAMERA_ISSUE
360                               battery backup          DEFECTIVE_CHARGER                       BATTERY_ISSUE
60 						battery backup very poor          DUPLICATE_PRODUCT                       BATTERY_ISSUE
704                    mobile hanging very often          OVERHEATING_ISSUE                       HANGING_ISSUE
707                     camera quality very poor          DUPLICATE_PRODUCT                        CAMERA_ISSUE
...

```


### Predict Return Comment

Given a (new) return comment, predict it's return category.

```python
>>> obj.predictReturn('The phone is hanging too much')
	HANGING_ISSUE

>>> obj.predictReturn('Music player not working properly :/')
	APPLICATIONS_NOT_WORKING

>>> obj.predictReturn('Sound not coming from jack')
	SPEAKER_ISSUE

>>> obj.predictReturn('Phone not starting up :(')
	DEVICE_POWER

>>> obj.predictReturn('I am very upset. Since morning i am not able to switch on device')
	DEVICE_POWER

>>> obj.predictReturn('Camera not functioning properly')
	CAMERA_ISSUE

>>> obj.predictReturn('Fingerprint sensor not working')
	FEATURE_RELATED
...
```

### Save Model

Save model for future use.

```python
>>> obj.save()
	Model save sucessful.
```






