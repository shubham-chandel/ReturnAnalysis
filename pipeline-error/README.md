# Pipeline Error

- The `return_sub_reasons` mentioned [here](https://docs.google.com/a/flipkart.com/spreadsheets/d/1zkcTsORgGc8Ll1mw8ZJsozpvzIvTUWlqzbSIi1fiSqE/edit?usp=sharing) and the one's saved to Bigfoot database `bigfoot_external_neo.scp_rrr__return_l2_id_level_hive_fact` differ by large numbers. 
- This repo helps in analysis of the same.

## Usage

```python
>>> from error import PipelineError
>>> obj = PipelineError(vertical="Handsets")
```

### Count non-listed reasons

Reasons generated on Bigfoot but not present in [oficial doc](https://docs.google.com/a/flipkart.com/spreadsheets/d/1zkcTsORgGc8Ll1mw8ZJsozpvzIvTUWlqzbSIi1fiSqE/edit?usp=sharing) for Handsets category.

```python
>>> obj.count()
	BULK_ORDER                            37688
	PURCHASED_MISTAKE                     21900
	LONG_DELIVERY_PROMISE                  6300
	DELIVERY_DELAYED                       5278
	UNAVAILABLE_AT_HOME                    5217
	EXPENSIVE_NOW                          4978
	PURCHASED_ELSEWHERE                    4088
	CHANGE_SHIP_ADDRESS                    1914
	BULK_ORDER_CANCEL                       820
	CANCELLED_DUE_TO_PICKUP_REATTEMPTS      783
	NON_SERVICEABLE                         384
	PRICE_ERROR                             162
	HEATING_ISSUE                            35
	CONNECTIVITY_SIGNAL_RELATED              29
	FAKE_ORDER_OPS                           28
	SIM_SLOT                                 24
	SENSORS_NOT_WORKING                      18
	USED_PRODUCT_SHIPPED                     17
	TECH_ERROR                               17
	PAN_NOT_AVAILABLE                        16
	AUDIO_SOUND_SPEAKER_RELATED              14
	...
```

### Plot Count vs Time graph

Plot shows count of not listed return reasons (Handsets) for the past year.

```python
>>> obj.plot()
```

![PLOT](/pipeline-error/error.png)


### Change Vertical

Change the vertical for further analysis as above.

```python
>>> obj.changeVertical('Books')
>>> obj.vertical
	Books
```



