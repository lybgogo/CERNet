from tensorflow.keras import layers, Model, Sequential, optimizers
import matplotlib.pyplot as plt
import random, csv
import tensorflow as tf
import json
from tensorflow.python.keras import regularizers
import PIL.Image as im
import numpy as np
from tensorflow import keras
from keras.layers import GlobalAveragePooling2D, GlobalMaxPooling2D, Reshape, Dense, multiply, Permute, Concatenate, \
    Conv2D, Add, Activation, Lambda
from keras import backend as K
from keras.activations import sigmoid
from tensorflow.python.keras.api._v2.keras import layers, Sequential, regularizers
from PIL import Image
import glob, os

import datetime

# In[2]:

begin_time = datetime.datetime.now()
BS = 32
num_class = 78
input_size = 224

#gpu加速
tf.config.set_soft_device_placement(True)
tf.debugging.set_log_device_placement(True)
gpus = tf.config.experimental.list_physical_devices('GPU')
tf.config.experimental.set_visible_devices(gpus[0], 'GPU')
tf.config.experimental.set_memory_growth(gpus[0], True)

train_image_path, val_image_path, test_image_path = [], [], []
image_path = glob.glob(r'../lyb_ear/train_in/*.png')  # 训练集
train_image_path = image_path[:int(0.6 * len(image_path))]
val_image_path = image_path[int(0.6 * len(image_path)):int(0.8 * len(image_path))]
test_image_path = image_path[int(0.8 * len(image_path)):]

print(len(train_image_path))
print(len(val_image_path))
print(len(test_image_path))

random.shuffle(train_image_path)
random.shuffle(val_image_path)
random.shuffle(test_image_path)

p_train_image_label, p_val_image_label, p_test_image_label = [], [], []
train_image_label, val_image_label, test_image_label = [], [], []

p_train_image_label = [(train_image_path[i].split('\\')[-1].split('_')[0]) for i in
                       range(len(train_image_path))]
for i in range(len(p_train_image_label)):
    Str = p_train_image_label[i]
    if (Str == '0') | (Str == '1') | (Str == '2') | (Str == '3')|(Str == '4') | (Str == '5') | (Str == '6') | (Str == '7')|(Str == '8') | (Str == '9'):
        train_image_label.append(0)
    elif (Str == '10') | (Str == '11') | (Str == '12') | (Str == '13')|(Str == '14') | (Str == '15') | (Str == '16') | (Str == '17')|(Str == '18') | (Str == '19'):
        train_image_label.append(1)
    elif (Str == '20') | (Str == '21') | (Str == '22') | (Str == '23')|(Str == '24') | (Str == '25') | (Str == '26') | (Str == '27')|(Str == '28') | (Str == '29'):
        train_image_label.append(2)
    elif (Str == '30') | (Str == '31') | (Str == '32') | (Str == '33')|(Str == '34') | (Str == '35') | (Str == '36') | (Str == '37')|(Str == '38') | (Str == '39'):
        train_image_label.append(3)
    elif (Str == '40') | (Str == '41') | (Str == '42') | (Str == '43')|(Str == '44') | (Str == '45') | (Str == '46') | (Str == '47')|(Str == '48') | (Str == '49'):
        train_image_label.append(4)
    elif (Str == '50') | (Str == '51') | (Str == '52') | (Str == '53')|(Str == '54') | (Str == '55') | (Str == '56') | (Str == '57')|(Str == '58') | (Str == '59'):
        train_image_label.append(5)
    elif (Str == '60') | (Str == '61') | (Str == '62') | (Str == '63')|(Str == '64') | (Str == '65') | (Str == '66') | (Str == '67')|(Str == '68') | (Str == '69'):
        train_image_label.append(6)
    elif (Str == '70') | (Str == '71') | (Str == '72') | (Str == '73')|(Str == '74') | (Str == '75') | (Str == '76') | (Str == '77')|(Str == '78') | (Str == '79'):
        train_image_label.append(7)
    elif (Str == '80') | (Str == '81') | (Str == '82') | (Str == '83')|(Str == '84') | (Str == '85') | (Str == '86') | (Str == '87')|(Str == '88') | (Str == '89'):
        train_image_label.append(8)
    elif (Str == '90') | (Str == '91') | (Str == '92') | (Str == '93')|(Str == '94') | (Str == '95') | (Str == '96') | (Str == '97')|(Str == '98') | (Str == '99'):
        train_image_label.append(9)
    elif (Str == '100') | (Str == '101') | (Str == '102') | (Str == '103')|(Str == '104') | (Str == '105') | (Str == '106') | (Str == '107')|(Str == '108') | (Str == '109'):
        train_image_label.append(10)
    elif (Str == '110') | (Str == '111') | (Str == '112') | (Str == '113')|(Str == '114') | (Str == '115') | (Str == '116') | (Str == '117')|(Str == '118') | (Str == '119'):
        train_image_label.append(11)
    elif (Str == '120') | (Str == '121') | (Str == '122') | (Str == '123')|(Str == '124') | (Str == '125') | (Str == '126') | (Str == '127')|(Str == '128') | (Str == '129'):
        train_image_label.append(12)
    elif (Str == '130') | (Str == '131') | (Str == '132') | (Str == '133')|(Str == '134') | (Str == '135') | (Str == '136') | (Str == '137')|(Str == '138') | (Str == '139'):
        train_image_label.append(13)
    elif (Str == '140') | (Str == '141') | (Str == '142') | (Str == '143')|(Str == '144') | (Str == '145') | (Str == '146') | (Str == '147')|(Str == '148') | (Str == '149'):
        train_image_label.append(14)
    elif (Str == '150') | (Str == '151') | (Str == '152') | (Str == '153')|(Str == '154') | (Str == '155') | (Str == '156') | (Str == '157')|(Str == '158') | (Str == '159'):
        train_image_label.append(15)
    elif (Str == '160') | (Str == '161') | (Str == '162') | (Str == '163')|(Str == '164') | (Str == '165') | (Str == '166') | (Str == '167')|(Str == '168') | (Str == '169'):
        train_image_label.append(16)
    elif (Str == '170') | (Str == '171') | (Str == '172') | (Str == '173')|(Str == '174') | (Str == '175') | (Str == '176') | (Str == '177')|(Str == '178') | (Str == '179'):
        train_image_label.append(17)
    elif (Str == '180') | (Str == '181') | (Str == '182') | (Str == '183')|(Str == '184') | (Str == '185') | (Str == '186') | (Str == '187')|(Str == '188') | (Str == '189'):
        train_image_label.append(18)
    elif (Str == '190') | (Str == '191') | (Str == '192') | (Str == '193')|(Str == '194') | (Str == '195') | (Str == '196') | (Str == '197')|(Str == '198') | (Str == '199'):
        train_image_label.append(19)
    elif (Str == '200') | (Str == '201') | (Str == '202') | (Str == '203')|(Str == '204') | (Str == '205') | (Str == '206') | (Str == '207')|(Str == '208') | (Str == '209'):
        train_image_label.append(20)
    elif (Str == '210') | (Str == '211') | (Str == '212') | (Str == '213')|(Str == '214') | (Str == '215') | (Str == '216') | (Str == '217')|(Str == '218') | (Str == '219'):
        train_image_label.append(21)
    elif (Str == '220') | (Str == '221') | (Str == '222') | (Str == '223')|(Str == '224') | (Str == '225') | (Str == '226') | (Str == '227')|(Str == '228') | (Str == '229'):
        train_image_label.append(22)
    elif (Str == '230') | (Str == '231') | (Str == '232') | (Str == '233')|(Str == '234') | (Str == '235') | (Str == '236') | (Str == '237')|(Str == '238') | (Str == '239'):
        train_image_label.append(23)
    elif (Str == '240') | (Str == '241') | (Str == '242') | (Str == '243')|(Str == '244') | (Str == '245') | (Str == '246') | (Str == '247')|(Str == '248') | (Str == '249'):
        train_image_label.append(24)
    elif (Str == '250') | (Str == '251') | (Str == '252') | (Str == '253')|(Str == '254') | (Str == '255') | (Str == '256') | (Str == '257')|(Str == '258') | (Str == '259'):
        train_image_label.append(25)
    elif (Str == '260') | (Str == '261') | (Str == '262') | (Str == '263')|(Str == '264') | (Str == '265') | (Str == '266') | (Str == '267')|(Str == '268') | (Str == '269'):
        train_image_label.append(26)
    elif (Str == '270') | (Str == '271') | (Str == '272') | (Str == '273')|(Str == '274') | (Str == '275') | (Str == '276') | (Str == '277')|(Str == '278') | (Str == '279'):
        train_image_label.append(27)
    elif (Str == '280') | (Str == '281') | (Str == '282') | (Str == '283')|(Str == '284') | (Str == '285') | (Str == '286') | (Str == '287')|(Str == '288') | (Str == '289'):
        train_image_label.append(28)
    elif (Str == '290') | (Str == '291') | (Str == '292') | (Str == '293')|(Str == '294') | (Str == '295') | (Str == '296') | (Str == '297')|(Str == '298') | (Str == '299'):
        train_image_label.append(29)
    elif (Str == '300') | (Str == '301') | (Str == '302') | (Str == '303')|(Str == '304') | (Str == '305') | (Str == '306') | (Str == '307')|(Str == '308') | (Str == '309'):
        train_image_label.append(30)
    elif (Str == '310') | (Str == '311') | (Str == '312') | (Str == '313')|(Str == '314') | (Str == '315') | (Str == '316') | (Str == '317')|(Str == '318') | (Str == '319'):
        train_image_label.append(31)
    elif (Str == '320') | (Str == '321') | (Str == '322') | (Str == '323')|(Str == '324') | (Str == '325') | (Str == '326') | (Str == '327')|(Str == '328') | (Str == '329'):
        train_image_label.append(32)
    elif (Str == '330') | (Str == '331') | (Str == '332') | (Str == '333')|(Str == '334') | (Str == '335') | (Str == '336') | (Str == '337')|(Str == '338') | (Str == '339'):
        train_image_label.append(33)
    elif (Str == '340') | (Str == '341') | (Str == '342') | (Str == '343')|(Str == '344') | (Str == '345') | (Str == '346') | (Str == '347')|(Str == '348') | (Str == '349'):
        train_image_label.append(34)
    elif (Str == '350') | (Str == '351') | (Str == '352') | (Str == '353')|(Str == '354') | (Str == '355') | (Str == '356') | (Str == '357')|(Str == '358') | (Str == '359'):
        train_image_label.append(35)
    elif (Str == '360') | (Str == '361') | (Str == '362') | (Str == '363')|(Str == '364') | (Str == '365') | (Str == '366') | (Str == '367')|(Str == '368') | (Str == '369'):
        train_image_label.append(36)
    elif (Str == '370') | (Str == '371') | (Str == '372') | (Str == '373')|(Str == '374') | (Str == '375') | (Str == '376') | (Str == '377')|(Str == '378') | (Str == '379'):
        train_image_label.append(37)
    elif (Str == '380') | (Str == '381') | (Str == '382') | (Str == '383')|(Str == '384') | (Str == '385') | (Str == '386') | (Str == '387')|(Str == '388') :
        train_image_label.append(38)
    elif (Str == '389') | (Str == '390') | (Str == '391') | (Str == '392')|(Str == '393') | (Str == '394') | (Str == '395') | (Str == '396')|(Str == '397') | (Str == '398'):
        train_image_label.append(39)
    elif (Str == '399') |(Str == '400') | (Str == '401') | (Str == '402') | (Str == '403')|(Str == '404') | (Str == '405') | (Str == '406') | (Str == '407')|(Str == '408') :
        train_image_label.append(40)
    elif (Str == '409') |(Str == '410') | (Str == '411') | (Str == '412') | (Str == '413')|(Str == '414') | (Str == '415') | (Str == '416') | (Str == '417')|(Str == '418') :
        train_image_label.append(41)
    elif (Str == '419') | (Str == '420') | (Str == '421') | (Str == '422') | (Str == '423')|(Str == '424') | (Str == '425') | (Str == '426') | (Str == '427')|(Str == '428') :
        train_image_label.append(42)
    elif (Str == '429') |(Str == '430') | (Str == '431') | (Str == '432') | (Str == '433')|(Str == '434') | (Str == '435') | (Str == '436') | (Str == '437')|(Str == '438') :
        train_image_label.append(43)
    elif (Str == '439') |(Str == '440') | (Str == '441') | (Str == '442') | (Str == '443')|(Str == '444') | (Str == '445') | (Str == '446') | (Str == '447')|(Str == '448') :
        train_image_label.append(44)
    elif (Str == '449') |(Str == '450') | (Str == '451') | (Str == '452') | (Str == '453')|(Str == '454') | (Str == '455') | (Str == '456') | (Str == '457')|(Str == '458') :
        train_image_label.append(45)
    elif (Str == '459') |(Str == '460') | (Str == '461') | (Str == '462') | (Str == '463')|(Str == '464') | (Str == '465') | (Str == '466') | (Str == '467')|(Str == '468')|(Str == '469') | (Str == '470') | (Str == '471') :
        train_image_label.append(46)
    elif (Str == '472') | (Str == '473')|(Str == '474') | (Str == '475') | (Str == '476') | (Str == '477')|(Str == '478') | (Str == '479') | (Str == '480') | (Str == '481'):
        train_image_label.append(47)
    elif (Str == '482') | (Str == '483')|(Str == '484') | (Str == '485') | (Str == '486') | (Str == '487')|(Str == '488') | (Str == '489')|(Str == '490') | (Str == '491') | (Str == '492') | (Str == '493')|(Str == '494'):
        train_image_label.append(48)
    elif (Str == '495') | (Str == '496') | (Str == '497')|(Str == '498') | (Str == '499')|(Str == '500') | (Str == '501') | (Str == '502') | (Str == '503')|(Str == '504'):
        train_image_label.append(49)
    elif (Str == '505') | (Str == '506') | (Str == '507')|(Str == '508') | (Str == '509')|(Str == '510') | (Str == '511') | (Str == '512') | (Str == '513')|(Str == '514'):
        train_image_label.append(50)
    elif (Str == '515') | (Str == '516') | (Str == '517')|(Str == '518') | (Str == '519')|(Str == '520') | (Str == '521') | (Str == '522') | (Str == '523')|(Str == '524'):
        train_image_label.append(51)
    elif (Str == '525') | (Str == '526') | (Str == '527')|(Str == '528') | (Str == '529')|(Str == '530') | (Str == '531') | (Str == '532') | (Str == '533')|(Str == '534'):
        train_image_label.append(52)
    elif (Str == '535') | (Str == '536') | (Str == '537')|(Str == '538') | (Str == '539')|(Str == '540') | (Str == '541') | (Str == '542') | (Str == '543')|(Str == '544'):
        train_image_label.append(53)
    elif (Str == '545') | (Str == '546') | (Str == '547')|(Str == '548') | (Str == '549')|(Str == '550') | (Str == '551') | (Str == '552') | (Str == '553')|(Str == '554'):
        train_image_label.append(54)
    elif (Str == '555') | (Str == '556') | (Str == '557')|(Str == '558') | (Str == '559')|(Str == '560') | (Str == '561') | (Str == '562') | (Str == '563')|(Str == '564'):
        train_image_label.append(55)
    elif (Str == '565') | (Str == '566') | (Str == '567')|(Str == '568') | (Str == '569')|(Str == '570') | (Str == '571') | (Str == '572') | (Str == '573')|(Str == '574'):
        train_image_label.append(56)
    elif (Str == '575') | (Str == '576') | (Str == '577')|(Str == '578') | (Str == '579')|(Str == '580') | (Str == '581') | (Str == '582') | (Str == '583')|(Str == '584'):
        train_image_label.append(57)
    elif (Str == '585') | (Str == '586') | (Str == '587')|(Str == '588') | (Str == '589')|(Str == '590') | (Str == '591') | (Str == '592') | (Str == '593')|(Str == '594'):
        train_image_label.append(58)
    elif (Str == '595') | (Str == '596') | (Str == '597')|(Str == '598') | (Str == '599')|(Str == '600') | (Str == '601') | (Str == '602') | (Str == '603')|(Str == '604'):
        train_image_label.append(59)
    elif (Str == '605') | (Str == '606') | (Str == '607')|(Str == '608') | (Str == '609')|(Str == '610') | (Str == '611') | (Str == '612') | (Str == '613')|(Str == '614'):
        train_image_label.append(60)
    elif (Str == '615') | (Str == '616') | (Str == '617')|(Str == '618') | (Str == '619')|(Str == '620') | (Str == '621') | (Str == '622') | (Str == '623')|(Str == '624'):
        train_image_label.append(61)
    elif (Str == '625') | (Str == '626') | (Str == '627')|(Str == '628') | (Str == '629')|(Str == '630') | (Str == '631') | (Str == '632') | (Str == '633')|(Str == '634'):
        train_image_label.append(62)
    elif (Str == '635') | (Str == '636') | (Str == '637')|(Str == '638') | (Str == '639')| (Str == '640') | (Str == '641') | (Str == '642') | (Str == '643')|(Str == '644'):
        train_image_label.append(63)
    elif (Str == '645') | (Str == '646') | (Str == '647')|(Str == '648') | (Str == '649')|(Str == '650') | (Str == '651') | (Str == '652') | (Str == '653')|(Str == '654') :
        train_image_label.append(64)
    elif (Str == '655') | (Str == '656') | (Str == '657')|(Str == '658') | (Str == '659')|(Str == '660') | (Str == '661') | (Str == '662') | (Str == '663')|(Str == '664'):
        train_image_label.append(65)
    elif (Str == '665') | (Str == '666') | (Str == '667')|(Str == '668') | (Str == '669')|(Str == '670') | (Str == '671') | (Str == '672') | (Str == '673')|(Str == '674'):
        train_image_label.append(66)
    elif (Str == '675') | (Str == '676') | (Str == '677')|(Str == '678') | (Str == '679')|(Str == '680') | (Str == '681') | (Str == '682') | (Str == '683')|(Str == '684'):
        train_image_label.append(67)
    elif (Str == '685') | (Str == '686') | (Str == '687')|(Str == '688') | (Str == '689')|(Str == '690') | (Str == '691') | (Str == '692') | (Str == '693')|(Str == '694'):
        train_image_label.append(68)
    elif (Str == '695') | (Str == '696') | (Str == '697')|(Str == '698') | (Str == '699')|(Str == '700') | (Str == '701') | (Str == '702') | (Str == '703')|(Str == '704'):
        train_image_label.append(69)
    elif (Str == '705') | (Str == '706') | (Str == '707')|(Str == '708') | (Str == '709')|(Str == '710') | (Str == '711') | (Str == '712') | (Str == '713')|(Str == '714'):
        train_image_label.append(70)
    elif (Str == '715') | (Str == '716') | (Str == '717')|(Str == '718') | (Str == '719')| (Str == '720') | (Str == '721') | (Str == '722') | (Str == '723')|(Str == '724'):
        train_image_label.append(71)
    elif (Str == '725') | (Str == '726') | (Str == '727')|(Str == '728') | (Str == '729')|(Str == '730') | (Str == '731') | (Str == '732') | (Str == '733')|(Str == '734'):
        train_image_label.append(72)
    elif (Str == '735') | (Str == '736') | (Str == '737')|(Str == '738') | (Str == '739')|(Str == '740') | (Str == '741') | (Str == '742') | (Str == '743')|(Str == '744'):
        train_image_label.append(73)
    elif (Str == '745') | (Str == '746') | (Str == '747')|(Str == '748') | (Str == '749')|(Str == '750') | (Str == '751') | (Str == '752') | (Str == '753')|(Str == '754'):
        train_image_label.append(74)
    elif (Str == '755') | (Str == '756') | (Str == '757')|(Str == '758') | (Str == '759')|(Str == '760') | (Str == '761') | (Str == '762') | (Str == '763')|(Str == '764'):
        train_image_label.append(75)
    elif (Str == '765') | (Str == '766') | (Str == '767') | (Str == '768')|(Str == '769') | (Str == '770') | (Str == '771') | (Str == '772')|(Str == '773') | (Str == '774'):
        train_image_label.append(76)
    elif (Str == '775') | (Str == '776') | (Str == '777') | (Str == '778')|(Str == '779') | (Str == '780') | (Str == '781') | (Str == '782')|(Str == '783') | (Str == '784'):
        train_image_label.append(77)
p_val_image_label = [(val_image_path[i].split('\\')[-1].split('_')[0]) for i in
                     range(len(val_image_path))]
for i in range(len(p_val_image_label)):
    Str = p_val_image_label[i]
    if (Str == '0') | (Str == '1') | (Str == '2') | (Str == '3') | (Str == '4') | (Str == '5') | (Str == '6') | (
            Str == '7') | (Str == '8') | (Str == '9'):
        val_image_label.append(0)
    elif (Str == '10') | (Str == '11') | (Str == '12') | (Str == '13') | (Str == '14') | (Str == '15') | (
            Str == '16') | (Str == '17') | (Str == '18') | (Str == '19'):
        val_image_label.append(1)
    elif (Str == '20') | (Str == '21') | (Str == '22') | (Str == '23') | (Str == '24') | (Str == '25') | (
            Str == '26') | (Str == '27') | (Str == '28') | (Str == '29'):
        val_image_label.append(2)
    elif (Str == '30') | (Str == '31') | (Str == '32') | (Str == '33') | (Str == '34') | (Str == '35') | (
            Str == '36') | (Str == '37') | (Str == '38') | (Str == '39'):
        val_image_label.append(3)
    elif (Str == '40') | (Str == '41') | (Str == '42') | (Str == '43') | (Str == '44') | (Str == '45') | (
            Str == '46') | (Str == '47') | (Str == '48') | (Str == '49'):
        val_image_label.append(4)
    elif (Str == '50') | (Str == '51') | (Str == '52') | (Str == '53') | (Str == '54') | (Str == '55') | (
            Str == '56') | (Str == '57') | (Str == '58') | (Str == '59'):
        val_image_label.append(5)
    elif (Str == '60') | (Str == '61') | (Str == '62') | (Str == '63') | (Str == '64') | (Str == '65') | (
            Str == '66') | (Str == '67') | (Str == '68') | (Str == '69'):
        val_image_label.append(6)
    elif (Str == '70') | (Str == '71') | (Str == '72') | (Str == '73') | (Str == '74') | (Str == '75') | (
            Str == '76') | (Str == '77') | (Str == '78') | (Str == '79'):
        val_image_label.append(7)
    elif (Str == '80') | (Str == '81') | (Str == '82') | (Str == '83') | (Str == '84') | (Str == '85') | (
            Str == '86') | (Str == '87') | (Str == '88') | (Str == '89'):
        val_image_label.append(8)
    elif (Str == '90') | (Str == '91') | (Str == '92') | (Str == '93') | (Str == '94') | (Str == '95') | (
            Str == '96') | (Str == '97') | (Str == '98') | (Str == '99'):
        val_image_label.append(9)
    elif (Str == '100') | (Str == '101') | (Str == '102') | (Str == '103') | (Str == '104') | (Str == '105') | (
            Str == '106') | (Str == '107') | (Str == '108') | (Str == '109'):
        val_image_label.append(10)
    elif (Str == '110') | (Str == '111') | (Str == '112') | (Str == '113') | (Str == '114') | (Str == '115') | (
            Str == '116') | (Str == '117') | (Str == '118') | (Str == '119'):
        val_image_label.append(11)
    elif (Str == '120') | (Str == '121') | (Str == '122') | (Str == '123') | (Str == '124') | (Str == '125') | (
            Str == '126') | (Str == '127') | (Str == '128') | (Str == '129'):
        val_image_label.append(12)
    elif (Str == '130') | (Str == '131') | (Str == '132') | (Str == '133') | (Str == '134') | (Str == '135') | (
            Str == '136') | (Str == '137') | (Str == '138') | (Str == '139'):
        val_image_label.append(13)
    elif (Str == '140') | (Str == '141') | (Str == '142') | (Str == '143') | (Str == '144') | (Str == '145') | (
            Str == '146') | (Str == '147') | (Str == '148') | (Str == '149'):
        val_image_label.append(14)
    elif (Str == '150') | (Str == '151') | (Str == '152') | (Str == '153') | (Str == '154') | (Str == '155') | (
            Str == '156') | (Str == '157') | (Str == '158') | (Str == '159'):
        val_image_label.append(15)
    elif (Str == '160') | (Str == '161') | (Str == '162') | (Str == '163') | (Str == '164') | (Str == '165') | (
            Str == '166') | (Str == '167') | (Str == '168') | (Str == '169'):
        val_image_label.append(16)
    elif (Str == '170') | (Str == '171') | (Str == '172') | (Str == '173') | (Str == '174') | (Str == '175') | (
            Str == '176') | (Str == '177') | (Str == '178') | (Str == '179'):
        val_image_label.append(17)
    elif (Str == '180') | (Str == '181') | (Str == '182') | (Str == '183') | (Str == '184') | (Str == '185') | (
            Str == '186') | (Str == '187') | (Str == '188') | (Str == '189'):
        val_image_label.append(18)
    elif (Str == '190') | (Str == '191') | (Str == '192') | (Str == '193') | (Str == '194') | (Str == '195') | (
            Str == '196') | (Str == '197') | (Str == '198') | (Str == '199'):
        val_image_label.append(19)
    elif (Str == '200') | (Str == '201') | (Str == '202') | (Str == '203') | (Str == '204') | (Str == '205') | (
            Str == '206') | (Str == '207') | (Str == '208') | (Str == '209'):
        val_image_label.append(20)
    elif (Str == '210') | (Str == '211') | (Str == '212') | (Str == '213') | (Str == '214') | (Str == '215') | (
            Str == '216') | (Str == '217') | (Str == '218') | (Str == '219'):
        val_image_label.append(21)
    elif (Str == '220') | (Str == '221') | (Str == '222') | (Str == '223') | (Str == '224') | (Str == '225') | (
            Str == '226') | (Str == '227') | (Str == '228') | (Str == '229'):
        val_image_label.append(22)
    elif (Str == '230') | (Str == '231') | (Str == '232') | (Str == '233') | (Str == '234') | (Str == '235') | (
            Str == '236') | (Str == '237') | (Str == '238') | (Str == '239'):
        val_image_label.append(23)
    elif (Str == '240') | (Str == '241') | (Str == '242') | (Str == '243') | (Str == '244') | (Str == '245') | (
            Str == '246') | (Str == '247') | (Str == '248') | (Str == '249'):
        val_image_label.append(24)
    elif (Str == '250') | (Str == '251') | (Str == '252') | (Str == '253') | (Str == '254') | (Str == '255') | (
            Str == '256') | (Str == '257') | (Str == '258') | (Str == '259'):
        val_image_label.append(25)
    elif (Str == '260') | (Str == '261') | (Str == '262') | (Str == '263') | (Str == '264') | (Str == '265') | (
            Str == '266') | (Str == '267') | (Str == '268') | (Str == '269'):
        val_image_label.append(26)
    elif (Str == '270') | (Str == '271') | (Str == '272') | (Str == '273') | (Str == '274') | (Str == '275') | (
            Str == '276') | (Str == '277') | (Str == '278') | (Str == '279'):
        val_image_label.append(27)
    elif (Str == '280') | (Str == '281') | (Str == '282') | (Str == '283') | (Str == '284') | (Str == '285') | (
            Str == '286') | (Str == '287') | (Str == '288') | (Str == '289'):
        val_image_label.append(28)
    elif (Str == '290') | (Str == '291') | (Str == '292') | (Str == '293') | (Str == '294') | (Str == '295') | (
            Str == '296') | (Str == '297') | (Str == '298') | (Str == '299'):
        val_image_label.append(29)
    elif (Str == '300') | (Str == '301') | (Str == '302') | (Str == '303') | (Str == '304') | (Str == '305') | (
            Str == '306') | (Str == '307') | (Str == '308') | (Str == '309'):
        val_image_label.append(30)
    elif (Str == '310') | (Str == '311') | (Str == '312') | (Str == '313') | (Str == '314') | (Str == '315') | (
            Str == '316') | (Str == '317') | (Str == '318') | (Str == '319'):
        val_image_label.append(31)
    elif (Str == '320') | (Str == '321') | (Str == '322') | (Str == '323') | (Str == '324') | (Str == '325') | (
            Str == '326') | (Str == '327') | (Str == '328') | (Str == '329'):
        val_image_label.append(32)
    elif (Str == '330') | (Str == '331') | (Str == '332') | (Str == '333') | (Str == '334') | (Str == '335') | (
            Str == '336') | (Str == '337') | (Str == '338') | (Str == '339'):
        val_image_label.append(33)
    elif (Str == '340') | (Str == '341') | (Str == '342') | (Str == '343') | (Str == '344') | (Str == '345') | (
            Str == '346') | (Str == '347') | (Str == '348') | (Str == '349'):
        val_image_label.append(34)
    elif (Str == '350') | (Str == '351') | (Str == '352') | (Str == '353') | (Str == '354') | (Str == '355') | (
            Str == '356') | (Str == '357') | (Str == '358') | (Str == '359'):
        val_image_label.append(35)
    elif (Str == '360') | (Str == '361') | (Str == '362') | (Str == '363') | (Str == '364') | (Str == '365') | (
            Str == '366') | (Str == '367') | (Str == '368') | (Str == '369'):
        val_image_label.append(36)
    elif (Str == '370') | (Str == '371') | (Str == '372') | (Str == '373') | (Str == '374') | (Str == '375') | (
            Str == '376') | (Str == '377') | (Str == '378') | (Str == '379'):
        val_image_label.append(37)
    elif (Str == '380') | (Str == '381') | (Str == '382') | (Str == '383') | (Str == '384') | (Str == '385') | (
            Str == '386') | (Str == '387') | (Str == '388'):
        val_image_label.append(38)
    elif (Str == '389') | (Str == '390') | (Str == '391') | (Str == '392') | (Str == '393') | (Str == '394') | (
            Str == '395') | (Str == '396') | (Str == '397') | (Str == '398'):
        val_image_label.append(39)
    elif (Str == '399') | (Str == '400') | (Str == '401') | (Str == '402') | (Str == '403') | (Str == '404') | (
            Str == '405') | (Str == '406') | (Str == '407') | (Str == '408'):
        val_image_label.append(40)
    elif (Str == '409') | (Str == '410') | (Str == '411') | (Str == '412') | (Str == '413') | (Str == '414') | (
            Str == '415') | (Str == '416') | (Str == '417') | (Str == '418'):
        val_image_label.append(41)
    elif (Str == '419') | (Str == '420') | (Str == '421') | (Str == '422') | (Str == '423') | (Str == '424') | (
            Str == '425') | (Str == '426') | (Str == '427') | (Str == '428'):
        val_image_label.append(42)
    elif (Str == '429') | (Str == '430') | (Str == '431') | (Str == '432') | (Str == '433') | (Str == '434') | (
            Str == '435') | (Str == '436') | (Str == '437') | (Str == '438'):
        val_image_label.append(43)
    elif (Str == '439') | (Str == '440') | (Str == '441') | (Str == '442') | (Str == '443') | (Str == '444') | (
            Str == '445') | (Str == '446') | (Str == '447') | (Str == '448'):
        val_image_label.append(44)
    elif (Str == '449') | (Str == '450') | (Str == '451') | (Str == '452') | (Str == '453') | (Str == '454') | (
            Str == '455') | (Str == '456') | (Str == '457') | (Str == '458'):
        val_image_label.append(45)
    elif (Str == '459') | (Str == '460') | (Str == '461') | (Str == '462') | (Str == '463') | (Str == '464') | (
            Str == '465') | (Str == '466') | (Str == '467') | (Str == '468') | (Str == '469') | (Str == '470') | (
            Str == '471'):
        val_image_label.append(46)
    elif (Str == '472') | (Str == '473') | (Str == '474') | (Str == '475') | (Str == '476') | (Str == '477') | (
            Str == '478') | (Str == '479') | (Str == '480') | (Str == '481'):
        val_image_label.append(47)
    elif (Str == '482') | (Str == '483') | (Str == '484') | (Str == '485') | (Str == '486') | (Str == '487') | (
            Str == '488') | (Str == '489') | (Str == '490') | (Str == '491') | (Str == '492') | (Str == '493') | (
            Str == '494'):
        val_image_label.append(48)
    elif (Str == '495') | (Str == '496') | (Str == '497') | (Str == '498') | (Str == '499') | (Str == '500') | (
            Str == '501') | (Str == '502') | (Str == '503') | (Str == '504'):
        val_image_label.append(49)
    elif (Str == '505') | (Str == '506') | (Str == '507') | (Str == '508') | (Str == '509') | (Str == '510') | (
            Str == '511') | (Str == '512') | (Str == '513') | (Str == '514'):
        val_image_label.append(50)
    elif (Str == '515') | (Str == '516') | (Str == '517') | (Str == '518') | (Str == '519') | (Str == '520') | (
            Str == '521') | (Str == '522') | (Str == '523') | (Str == '524'):
        val_image_label.append(51)
    elif (Str == '525') | (Str == '526') | (Str == '527') | (Str == '528') | (Str == '529') | (Str == '530') | (
            Str == '531') | (Str == '532') | (Str == '533') | (Str == '534'):
        val_image_label.append(52)
    elif (Str == '535') | (Str == '536') | (Str == '537') | (Str == '538') | (Str == '539') | (Str == '540') | (
            Str == '541') | (Str == '542') | (Str == '543') | (Str == '544'):
        val_image_label.append(53)
    elif (Str == '545') | (Str == '546') | (Str == '547') | (Str == '548') | (Str == '549') | (Str == '550') | (
            Str == '551') | (Str == '552') | (Str == '553') | (Str == '554'):
        val_image_label.append(54)
    elif (Str == '555') | (Str == '556') | (Str == '557') | (Str == '558') | (Str == '559') | (Str == '560') | (
            Str == '561') | (Str == '562') | (Str == '563') | (Str == '564'):
        val_image_label.append(55)
    elif (Str == '565') | (Str == '566') | (Str == '567') | (Str == '568') | (Str == '569') | (Str == '570') | (
            Str == '571') | (Str == '572') | (Str == '573') | (Str == '574'):
        val_image_label.append(56)
    elif (Str == '575') | (Str == '576') | (Str == '577') | (Str == '578') | (Str == '579') | (Str == '580') | (
            Str == '581') | (Str == '582') | (Str == '583') | (Str == '584'):
        val_image_label.append(57)
    elif (Str == '585') | (Str == '586') | (Str == '587') | (Str == '588') | (Str == '589') | (Str == '590') | (
            Str == '591') | (Str == '592') | (Str == '593') | (Str == '594'):
        val_image_label.append(58)
    elif (Str == '595') | (Str == '596') | (Str == '597') | (Str == '598') | (Str == '599') | (Str == '600') | (
            Str == '601') | (Str == '602') | (Str == '603') | (Str == '604'):
        val_image_label.append(59)
    elif (Str == '605') | (Str == '606') | (Str == '607') | (Str == '608') | (Str == '609') | (Str == '610') | (
            Str == '611') | (Str == '612') | (Str == '613') | (Str == '614'):
        val_image_label.append(60)
    elif (Str == '615') | (Str == '616') | (Str == '617') | (Str == '618') | (Str == '619') | (Str == '620') | (
            Str == '621') | (Str == '622') | (Str == '623') | (Str == '624'):
        val_image_label.append(61)
    elif (Str == '625') | (Str == '626') | (Str == '627') | (Str == '628') | (Str == '629') | (Str == '630') | (
            Str == '631') | (Str == '632') | (Str == '633') | (Str == '634'):
        val_image_label.append(62)
    elif (Str == '635') | (Str == '636') | (Str == '637') | (Str == '638') | (Str == '639') | (Str == '640') | (
            Str == '641') | (Str == '642') | (Str == '643') | (Str == '644'):
        val_image_label.append(63)
    elif (Str == '645') | (Str == '646') | (Str == '647') | (Str == '648') | (Str == '649') | (Str == '650') | (
            Str == '651') | (Str == '652') | (Str == '653') | (Str == '654'):
        val_image_label.append(64)
    elif (Str == '655') | (Str == '656') | (Str == '657') | (Str == '658') | (Str == '659') | (Str == '660') | (
            Str == '661') | (Str == '662') | (Str == '663') | (Str == '664'):
        val_image_label.append(65)
    elif (Str == '665') | (Str == '666') | (Str == '667') | (Str == '668') | (Str == '669') | (Str == '670') | (
            Str == '671') | (Str == '672') | (Str == '673') | (Str == '674'):
        val_image_label.append(66)
    elif (Str == '675') | (Str == '676') | (Str == '677') | (Str == '678') | (Str == '679') | (Str == '680') | (
            Str == '681') | (Str == '682') | (Str == '683') | (Str == '684'):
        val_image_label.append(67)
    elif (Str == '685') | (Str == '686') | (Str == '687') | (Str == '688') | (Str == '689') | (Str == '690') | (
            Str == '691') | (Str == '692') | (Str == '693') | (Str == '694'):
        val_image_label.append(68)
    elif (Str == '695') | (Str == '696') | (Str == '697') | (Str == '698') | (Str == '699') | (Str == '700') | (
            Str == '701') | (Str == '702') | (Str == '703') | (Str == '704'):
        val_image_label.append(69)
    elif (Str == '705') | (Str == '706') | (Str == '707') | (Str == '708') | (Str == '709') | (Str == '710') | (
            Str == '711') | (Str == '712') | (Str == '713') | (Str == '714'):
        val_image_label.append(70)
    elif (Str == '715') | (Str == '716') | (Str == '717') | (Str == '718') | (Str == '719') | (Str == '720') | (
            Str == '721') | (Str == '722') | (Str == '723') | (Str == '724'):
        val_image_label.append(71)
    elif (Str == '725') | (Str == '726') | (Str == '727') | (Str == '728') | (Str == '729') | (Str == '730') | (
            Str == '731') | (Str == '732') | (Str == '733') | (Str == '734'):
        val_image_label.append(72)
    elif (Str == '735') | (Str == '736') | (Str == '737') | (Str == '738') | (Str == '739') | (Str == '740') | (
            Str == '741') | (Str == '742') | (Str == '743') | (Str == '744'):
        val_image_label.append(73)
    elif (Str == '745') | (Str == '746') | (Str == '747') | (Str == '748') | (Str == '749') | (Str == '750') | (
            Str == '751') | (Str == '752') | (Str == '753') | (Str == '754'):
        val_image_label.append(74)
    elif (Str == '755') | (Str == '756') | (Str == '757') | (Str == '758') | (Str == '759') | (Str == '760') | (
            Str == '761') | (Str == '762') | (Str == '763') | (Str == '764'):
        val_image_label.append(75)
    elif (Str == '765') | (Str == '766') | (Str == '767') | (Str == '768') | (Str == '769') | (Str == '770') | (
            Str == '771') | (Str == '772') | (Str == '773') | (Str == '774'):
        val_image_label.append(76)
    elif (Str == '775') | (Str == '776') | (Str == '777') | (Str == '778') | (Str == '779') | (Str == '780') | (
            Str == '781') | (Str == '782') | (Str == '783') | (Str == '784'):
        val_image_label.append(77)

p_test_image_label = [(test_image_path[i].split('\\')[-1].split('_')[0]) for i in
                      range(len(test_image_path))]

for i in range(len(p_test_image_label)):
    Str = p_test_image_label[i]
    if (Str == '0') | (Str == '1') | (Str == '2') | (Str == '3') | (Str == '4') | (Str == '5') | (Str == '6') | (
            Str == '7') | (Str == '8') | (Str == '9'):
        test_image_label.append(0)
    elif (Str == '10') | (Str == '11') | (Str == '12') | (Str == '13') | (Str == '14') | (Str == '15') | (
            Str == '16') | (Str == '17') | (Str == '18') | (Str == '19'):
        test_image_label.append(1)
    elif (Str == '20') | (Str == '21') | (Str == '22') | (Str == '23') | (Str == '24') | (Str == '25') | (
            Str == '26') | (Str == '27') | (Str == '28') | (Str == '29'):
        test_image_label.append(2)
    elif (Str == '30') | (Str == '31') | (Str == '32') | (Str == '33') | (Str == '34') | (Str == '35') | (
            Str == '36') | (Str == '37') | (Str == '38') | (Str == '39'):
        test_image_label.append(3)
    elif (Str == '40') | (Str == '41') | (Str == '42') | (Str == '43') | (Str == '44') | (Str == '45') | (
            Str == '46') | (Str == '47') | (Str == '48') | (Str == '49'):
        test_image_label.append(4)
    elif (Str == '50') | (Str == '51') | (Str == '52') | (Str == '53') | (Str == '54') | (Str == '55') | (
            Str == '56') | (Str == '57') | (Str == '58') | (Str == '59'):
        test_image_label.append(5)
    elif (Str == '60') | (Str == '61') | (Str == '62') | (Str == '63') | (Str == '64') | (Str == '65') | (
            Str == '66') | (Str == '67') | (Str == '68') | (Str == '69'):
        test_image_label.append(6)
    elif (Str == '70') | (Str == '71') | (Str == '72') | (Str == '73') | (Str == '74') | (Str == '75') | (
            Str == '76') | (Str == '77') | (Str == '78') | (Str == '79'):
        test_image_label.append(7)
    elif (Str == '80') | (Str == '81') | (Str == '82') | (Str == '83') | (Str == '84') | (Str == '85') | (
            Str == '86') | (Str == '87') | (Str == '88') | (Str == '89'):
        test_image_label.append(8)
    elif (Str == '90') | (Str == '91') | (Str == '92') | (Str == '93') | (Str == '94') | (Str == '95') | (
            Str == '96') | (Str == '97') | (Str == '98') | (Str == '99'):
        test_image_label.append(9)
    elif (Str == '100') | (Str == '101') | (Str == '102') | (Str == '103') | (Str == '104') | (Str == '105') | (
            Str == '106') | (Str == '107') | (Str == '108') | (Str == '109'):
        test_image_label.append(10)
    elif (Str == '110') | (Str == '111') | (Str == '112') | (Str == '113') | (Str == '114') | (Str == '115') | (
            Str == '116') | (Str == '117') | (Str == '118') | (Str == '119'):
        test_image_label.append(11)
    elif (Str == '120') | (Str == '121') | (Str == '122') | (Str == '123') | (Str == '124') | (Str == '125') | (
            Str == '126') | (Str == '127') | (Str == '128') | (Str == '129'):
        test_image_label.append(12)
    elif (Str == '130') | (Str == '131') | (Str == '132') | (Str == '133') | (Str == '134') | (Str == '135') | (
            Str == '136') | (Str == '137') | (Str == '138') | (Str == '139'):
        test_image_label.append(13)
    elif (Str == '140') | (Str == '141') | (Str == '142') | (Str == '143') | (Str == '144') | (Str == '145') | (
            Str == '146') | (Str == '147') | (Str == '148') | (Str == '149'):
        test_image_label.append(14)
    elif (Str == '150') | (Str == '151') | (Str == '152') | (Str == '153') | (Str == '154') | (Str == '155') | (
            Str == '156') | (Str == '157') | (Str == '158') | (Str == '159'):
        test_image_label.append(15)
    elif (Str == '160') | (Str == '161') | (Str == '162') | (Str == '163') | (Str == '164') | (Str == '165') | (
            Str == '166') | (Str == '167') | (Str == '168') | (Str == '169'):
        test_image_label.append(16)
    elif (Str == '170') | (Str == '171') | (Str == '172') | (Str == '173') | (Str == '174') | (Str == '175') | (
            Str == '176') | (Str == '177') | (Str == '178') | (Str == '179'):
        test_image_label.append(17)
    elif (Str == '180') | (Str == '181') | (Str == '182') | (Str == '183') | (Str == '184') | (Str == '185') | (
            Str == '186') | (Str == '187') | (Str == '188') | (Str == '189'):
        test_image_label.append(18)
    elif (Str == '190') | (Str == '191') | (Str == '192') | (Str == '193') | (Str == '194') | (Str == '195') | (
            Str == '196') | (Str == '197') | (Str == '198') | (Str == '199'):
        test_image_label.append(19)
    elif (Str == '200') | (Str == '201') | (Str == '202') | (Str == '203') | (Str == '204') | (Str == '205') | (
            Str == '206') | (Str == '207') | (Str == '208') | (Str == '209'):
        test_image_label.append(20)
    elif (Str == '210') | (Str == '211') | (Str == '212') | (Str == '213') | (Str == '214') | (Str == '215') | (
            Str == '216') | (Str == '217') | (Str == '218') | (Str == '219'):
        test_image_label.append(21)
    elif (Str == '220') | (Str == '221') | (Str == '222') | (Str == '223') | (Str == '224') | (Str == '225') | (
            Str == '226') | (Str == '227') | (Str == '228') | (Str == '229'):
        test_image_label.append(22)
    elif (Str == '230') | (Str == '231') | (Str == '232') | (Str == '233') | (Str == '234') | (Str == '235') | (
            Str == '236') | (Str == '237') | (Str == '238') | (Str == '239'):
        test_image_label.append(23)
    elif (Str == '240') | (Str == '241') | (Str == '242') | (Str == '243') | (Str == '244') | (Str == '245') | (
            Str == '246') | (Str == '247') | (Str == '248') | (Str == '249'):
        test_image_label.append(24)
    elif (Str == '250') | (Str == '251') | (Str == '252') | (Str == '253') | (Str == '254') | (Str == '255') | (
            Str == '256') | (Str == '257') | (Str == '258') | (Str == '259'):
        test_image_label.append(25)
    elif (Str == '260') | (Str == '261') | (Str == '262') | (Str == '263') | (Str == '264') | (Str == '265') | (
            Str == '266') | (Str == '267') | (Str == '268') | (Str == '269'):
        test_image_label.append(26)
    elif (Str == '270') | (Str == '271') | (Str == '272') | (Str == '273') | (Str == '274') | (Str == '275') | (
            Str == '276') | (Str == '277') | (Str == '278') | (Str == '279'):
        test_image_label.append(27)
    elif (Str == '280') | (Str == '281') | (Str == '282') | (Str == '283') | (Str == '284') | (Str == '285') | (
            Str == '286') | (Str == '287') | (Str == '288') | (Str == '289'):
        test_image_label.append(28)
    elif (Str == '290') | (Str == '291') | (Str == '292') | (Str == '293') | (Str == '294') | (Str == '295') | (
            Str == '296') | (Str == '297') | (Str == '298') | (Str == '299'):
        test_image_label.append(29)
    elif (Str == '300') | (Str == '301') | (Str == '302') | (Str == '303') | (Str == '304') | (Str == '305') | (
            Str == '306') | (Str == '307') | (Str == '308') | (Str == '309'):
        test_image_label.append(30)
    elif (Str == '310') | (Str == '311') | (Str == '312') | (Str == '313') | (Str == '314') | (Str == '315') | (
            Str == '316') | (Str == '317') | (Str == '318') | (Str == '319'):
        test_image_label.append(31)
    elif (Str == '320') | (Str == '321') | (Str == '322') | (Str == '323') | (Str == '324') | (Str == '325') | (
            Str == '326') | (Str == '327') | (Str == '328') | (Str == '329'):
        test_image_label.append(32)
    elif (Str == '330') | (Str == '331') | (Str == '332') | (Str == '333') | (Str == '334') | (Str == '335') | (
            Str == '336') | (Str == '337') | (Str == '338') | (Str == '339'):
        test_image_label.append(33)
    elif (Str == '340') | (Str == '341') | (Str == '342') | (Str == '343') | (Str == '344') | (Str == '345') | (
            Str == '346') | (Str == '347') | (Str == '348') | (Str == '349'):
        test_image_label.append(34)
    elif (Str == '350') | (Str == '351') | (Str == '352') | (Str == '353') | (Str == '354') | (Str == '355') | (
            Str == '356') | (Str == '357') | (Str == '358') | (Str == '359'):
        test_image_label.append(35)
    elif (Str == '360') | (Str == '361') | (Str == '362') | (Str == '363') | (Str == '364') | (Str == '365') | (
            Str == '366') | (Str == '367') | (Str == '368') | (Str == '369'):
        test_image_label.append(36)
    elif (Str == '370') | (Str == '371') | (Str == '372') | (Str == '373') | (Str == '374') | (Str == '375') | (
            Str == '376') | (Str == '377') | (Str == '378') | (Str == '379'):
        test_image_label.append(37)
    elif (Str == '380') | (Str == '381') | (Str == '382') | (Str == '383') | (Str == '384') | (Str == '385') | (
            Str == '386') | (Str == '387') | (Str == '388'):
        test_image_label.append(38)
    elif (Str == '389') | (Str == '390') | (Str == '391') | (Str == '392') | (Str == '393') | (Str == '394') | (
            Str == '395') | (Str == '396') | (Str == '397') | (Str == '398'):
        test_image_label.append(39)
    elif (Str == '399') | (Str == '400') | (Str == '401') | (Str == '402') | (Str == '403') | (Str == '404') | (
            Str == '405') | (Str == '406') | (Str == '407') | (Str == '408'):
        test_image_label.append(40)
    elif (Str == '409') | (Str == '410') | (Str == '411') | (Str == '412') | (Str == '413') | (Str == '414') | (
            Str == '415') | (Str == '416') | (Str == '417') | (Str == '418'):
        test_image_label.append(41)
    elif (Str == '419') | (Str == '420') | (Str == '421') | (Str == '422') | (Str == '423') | (Str == '424') | (
            Str == '425') | (Str == '426') | (Str == '427') | (Str == '428'):
        test_image_label.append(42)
    elif (Str == '429') | (Str == '430') | (Str == '431') | (Str == '432') | (Str == '433') | (Str == '434') | (
            Str == '435') | (Str == '436') | (Str == '437') | (Str == '438'):
        test_image_label.append(43)
    elif (Str == '439') | (Str == '440') | (Str == '441') | (Str == '442') | (Str == '443') | (Str == '444') | (
            Str == '445') | (Str == '446') | (Str == '447') | (Str == '448'):
        test_image_label.append(44)
    elif (Str == '449') | (Str == '450') | (Str == '451') | (Str == '452') | (Str == '453') | (Str == '454') | (
            Str == '455') | (Str == '456') | (Str == '457') | (Str == '458'):
        test_image_label.append(45)
    elif (Str == '459') | (Str == '460') | (Str == '461') | (Str == '462') | (Str == '463') | (Str == '464') | (
            Str == '465') | (Str == '466') | (Str == '467') | (Str == '468') | (Str == '469') | (Str == '470') | (
            Str == '471'):
        test_image_label.append(46)
    elif (Str == '472') | (Str == '473') | (Str == '474') | (Str == '475') | (Str == '476') | (Str == '477') | (
            Str == '478') | (Str == '479') | (Str == '480') | (Str == '481'):
        test_image_label.append(47)
    elif (Str == '482') | (Str == '483') | (Str == '484') | (Str == '485') | (Str == '486') | (Str == '487') | (
            Str == '488') | (Str == '489') | (Str == '490') | (Str == '491') | (Str == '492') | (Str == '493') | (
            Str == '494'):
        test_image_label.append(48)
    elif (Str == '495') | (Str == '496') | (Str == '497') | (Str == '498') | (Str == '499') | (Str == '500') | (
            Str == '501') | (Str == '502') | (Str == '503') | (Str == '504'):
        test_image_label.append(49)
    elif (Str == '505') | (Str == '506') | (Str == '507') | (Str == '508') | (Str == '509') | (Str == '510') | (
            Str == '511') | (Str == '512') | (Str == '513') | (Str == '514'):
        test_image_label.append(50)
    elif (Str == '515') | (Str == '516') | (Str == '517') | (Str == '518') | (Str == '519') | (Str == '520') | (
            Str == '521') | (Str == '522') | (Str == '523') | (Str == '524'):
        test_image_label.append(51)
    elif (Str == '525') | (Str == '526') | (Str == '527') | (Str == '528') | (Str == '529') | (Str == '530') | (
            Str == '531') | (Str == '532') | (Str == '533') | (Str == '534'):
        test_image_label.append(52)
    elif (Str == '535') | (Str == '536') | (Str == '537') | (Str == '538') | (Str == '539') | (Str == '540') | (
            Str == '541') | (Str == '542') | (Str == '543') | (Str == '544'):
        test_image_label.append(53)
    elif (Str == '545') | (Str == '546') | (Str == '547') | (Str == '548') | (Str == '549') | (Str == '550') | (
            Str == '551') | (Str == '552') | (Str == '553') | (Str == '554'):
        test_image_label.append(54)
    elif (Str == '555') | (Str == '556') | (Str == '557') | (Str == '558') | (Str == '559') | (Str == '560') | (
            Str == '561') | (Str == '562') | (Str == '563') | (Str == '564'):
        test_image_label.append(55)
    elif (Str == '565') | (Str == '566') | (Str == '567') | (Str == '568') | (Str == '569') | (Str == '570') | (
            Str == '571') | (Str == '572') | (Str == '573') | (Str == '574'):
        test_image_label.append(56)
    elif (Str == '575') | (Str == '576') | (Str == '577') | (Str == '578') | (Str == '579') | (Str == '580') | (
            Str == '581') | (Str == '582') | (Str == '583') | (Str == '584'):
        test_image_label.append(57)
    elif (Str == '585') | (Str == '586') | (Str == '587') | (Str == '588') | (Str == '589') | (Str == '590') | (
            Str == '591') | (Str == '592') | (Str == '593') | (Str == '594'):
        test_image_label.append(58)
    elif (Str == '595') | (Str == '596') | (Str == '597') | (Str == '598') | (Str == '599') | (Str == '600') | (
            Str == '601') | (Str == '602') | (Str == '603') | (Str == '604'):
        test_image_label.append(59)
    elif (Str == '605') | (Str == '606') | (Str == '607') | (Str == '608') | (Str == '609') | (Str == '610') | (
            Str == '611') | (Str == '612') | (Str == '613') | (Str == '614'):
        test_image_label.append(60)
    elif (Str == '615') | (Str == '616') | (Str == '617') | (Str == '618') | (Str == '619') | (Str == '620') | (
            Str == '621') | (Str == '622') | (Str == '623') | (Str == '624'):
        test_image_label.append(61)
    elif (Str == '625') | (Str == '626') | (Str == '627') | (Str == '628') | (Str == '629') | (Str == '630') | (
            Str == '631') | (Str == '632') | (Str == '633') | (Str == '634'):
        test_image_label.append(62)
    elif (Str == '635') | (Str == '636') | (Str == '637') | (Str == '638') | (Str == '639') | (Str == '640') | (
            Str == '641') | (Str == '642') | (Str == '643') | (Str == '644'):
        test_image_label.append(63)
    elif (Str == '645') | (Str == '646') | (Str == '647') | (Str == '648') | (Str == '649') | (Str == '650') | (
            Str == '651') | (Str == '652') | (Str == '653') | (Str == '654'):
        test_image_label.append(64)
    elif (Str == '655') | (Str == '656') | (Str == '657') | (Str == '658') | (Str == '659') | (Str == '660') | (
            Str == '661') | (Str == '662') | (Str == '663') | (Str == '664'):
        test_image_label.append(65)
    elif (Str == '665') | (Str == '666') | (Str == '667') | (Str == '668') | (Str == '669') | (Str == '670') | (
            Str == '671') | (Str == '672') | (Str == '673') | (Str == '674'):
        test_image_label.append(66)
    elif (Str == '675') | (Str == '676') | (Str == '677') | (Str == '678') | (Str == '679') | (Str == '680') | (
            Str == '681') | (Str == '682') | (Str == '683') | (Str == '684'):
        test_image_label.append(67)
    elif (Str == '685') | (Str == '686') | (Str == '687') | (Str == '688') | (Str == '689') | (Str == '690') | (
            Str == '691') | (Str == '692') | (Str == '693') | (Str == '694'):
        test_image_label.append(68)
    elif (Str == '695') | (Str == '696') | (Str == '697') | (Str == '698') | (Str == '699') | (Str == '700') | (
            Str == '701') | (Str == '702') | (Str == '703') | (Str == '704'):
        test_image_label.append(69)
    elif (Str == '705') | (Str == '706') | (Str == '707') | (Str == '708') | (Str == '709') | (Str == '710') | (
            Str == '711') | (Str == '712') | (Str == '713') | (Str == '714'):
        test_image_label.append(70)
    elif (Str == '715') | (Str == '716') | (Str == '717') | (Str == '718') | (Str == '719') | (Str == '720') | (
            Str == '721') | (Str == '722') | (Str == '723') | (Str == '724'):
        test_image_label.append(71)
    elif (Str == '725') | (Str == '726') | (Str == '727') | (Str == '728') | (Str == '729') | (Str == '730') | (
            Str == '731') | (Str == '732') | (Str == '733') | (Str == '734'):
        test_image_label.append(72)
    elif (Str == '735') | (Str == '736') | (Str == '737') | (Str == '738') | (Str == '739') | (Str == '740') | (
            Str == '741') | (Str == '742') | (Str == '743') | (Str == '744'):
        test_image_label.append(73)
    elif (Str == '745') | (Str == '746') | (Str == '747') | (Str == '748') | (Str == '749') | (Str == '750') | (
            Str == '751') | (Str == '752') | (Str == '753') | (Str == '754'):
        test_image_label.append(74)
    elif (Str == '755') | (Str == '756') | (Str == '757') | (Str == '758') | (Str == '759') | (Str == '760') | (
            Str == '761') | (Str == '762') | (Str == '763') | (Str == '764'):
        test_image_label.append(75)
    elif (Str == '765') | (Str == '766') | (Str == '767') | (Str == '768') | (Str == '769') | (Str == '770') | (
            Str == '771') | (Str == '772') | (Str == '773') | (Str == '774'):
        test_image_label.append(76)
    elif (Str == '775') | (Str == '776') | (Str == '777') | (Str == '778') | (Str == '779') | (Str == '780') | (
            Str == '781') | (Str == '782') | (Str == '783') | (Str == '784'):
        test_image_label.append(77)
print(len(train_image_label))
print(len(val_image_label))
print(len(test_image_label))


img_mean = tf.constant([0.485, 0.456, 0.406])
img_std = tf.constant([0.229, 0.224, 0.225])


def normalize(x, mean=img_mean, std=img_std):
    x = (x - mean) / std
    return x

def preprocess(image_path, label):
    x = tf.io.read_file(image_path)
    x = tf.image.decode_png(x, channels=3)
    x = tf.image.resize(x, [input_size, input_size])
    # x = tf.image.random_brightness(x, max_delta=0.5)#调整亮度
    # x = tf.image.random_contrast(x, 0.1, 0.6)#调整对比度
    # x = tf.image.random_hue(x, 0.5)#调整图片色相
    # x = tf.image.random_saturation(x, 0, 5)#调整图片饱和度
    x = tf.image.random_crop(x, [input_size, input_size, 3])  # 裁剪
    x = tf.cast(x, dtype=tf.float32) / 255.
    x = normalize(x)
    y = tf.convert_to_tensor(label)
    return x, y

train_images, train_labels = [], []
train_images = train_image_path
train_labels = np.array(train_image_label)
train_labels = train_labels.reshape(-1, 1)
train_labels = tf.squeeze(train_labels, axis=1)
train_labels = tf.one_hot(train_labels, depth=num_class).numpy()
train_db = tf.data.Dataset.from_tensor_slices((train_images, train_labels))  # images: string path， labels: number
train_db = train_db.shuffle(len(train_image_path)).map(preprocess).batch(BS)

val_images, val_labels = [], []
val_images = val_image_path
val_labels = np.array(val_image_label)
val_labels = val_labels.reshape(-1, 1)
val_labels = tf.squeeze(val_labels, axis=1)
val_labels = tf.one_hot(val_labels, depth=num_class).numpy()
val_db = tf.data.Dataset.from_tensor_slices((val_images, val_labels))  # images: string path， labels: number
val_db = val_db.shuffle(len(val_image_label)).map(preprocess).batch(BS)


test_images, test_labels = [], []
test_images = test_image_path
test_labels = np.array(test_image_label)
test_labels = test_labels.reshape(-1, 1)
test_labels = tf.squeeze(test_labels, axis=1)
test_labels = tf.one_hot(test_labels, depth=num_class).numpy()
test_db = tf.data.Dataset.from_tensor_slices((test_images, test_labels))  # images: string path， labels: number
test_db = test_db.shuffle(len(test_image_path)).map(preprocess).batch(BS)

print(len(train_image_label))
print(len(val_image_label))
print(len(test_image_label))

# 瓶颈层，相当于每一个稠密块中若干个相同的H函数  +SE  改进
# BN -> ReLU -> 1*1 Conv -> BN -> ReLU -> 3*3 Conv
class BottleNeck(layers.Layer):
    # growth_rate对应的是论文中的增长率k，指经过一个BottleNet输出的特征图的通道数；drop_rate指失活率。
    def __init__(self, growth_rate, drop_rate):
        super(BottleNeck, self).__init__()
        self.bn1 = layers.BatchNormalization()
        self.conv1 = layers.Conv2D(filters=4 * growth_rate,  # 使用1*1卷积核将通道数降维到4*k
                                   kernel_size=(1, 1),
                                   strides=1,
                                   padding="same")
        self.bn2 = layers.BatchNormalization()
        self.conv2 = layers.Conv2D(filters=growth_rate,  # 使用3*3卷积核，使得输出维度（通道数）为k
                                   kernel_size=(3, 3),
                                   strides=1,
                                   padding="same")
        self.dropout = layers.Dropout(rate=drop_rate)

        #         SE-block
        self.se_gap = layers.GlobalAveragePooling2D()
        self.se_avgp = layers.AveragePooling2D(pool_size=3, strides=3)
        # self.se_avgp = layers.MaxPooling2D(pool_size=3, strides=3)
        self.se_resize = layers.Reshape((1, 1, growth_rate))
        self.se_fc1 = layers.Dense(units=growth_rate // 16, activation=tf.keras.activations.relu)
        self.se_fc2 = layers.Dense(units=growth_rate, activation=tf.keras.activations.sigmoid)

        # 将网络层存入一个列表中
        self.listLayers = [self.bn1,
                           layers.Activation("relu"),
                           self.conv1,
                           self.bn2,
                           layers.Activation("relu"),
                           self.conv2,
                           self.dropout]

    def call(self, x):
        y = x
        for layer in self.listLayers.layers:
            y = layer(y)

        # 全局平均池化
        b = self.se_gap(y)
        b = self.se_resize(b)

        # 局部平均池化
        c = y
        c = self.se_avgp(c)
        c = layers.Reshape((1, 1, c.shape[1] * c.shape[2] * c.shape[3]))(c)

        d = tf.concat([b, c], axis=-1)

        d = self.se_fc1(d)
        d = self.se_fc2(d)
        y = layers.Multiply()([y, d])
        # 每经过一个BottleNet，将输入和输出按通道连结。作用是：将前l层的输入连结起来，作为下一个BottleNet的输入。
        y = layers.concatenate([x, y], axis=-1)  # 第一次3+...
        return y


# 稠密块，由若干个相同的瓶颈层构成
# BottleNeck * 6
class DenseBlock(layers.Layer):
    # num_layers表示该稠密块存在BottleNet的个数，也就是一个稠密块的层数L
    # 121为：6， 12， 24， 16
    def __init__(self, num_layers, growth_rate, drop_rate=0.5):
        super(DenseBlock, self).__init__()
        self.num_layers = num_layers
        self.growth_rate = growth_rate
        self.drop_rate = drop_rate
        self.listLayers = []
        # 一个DenseBlock由多个相同的BottleNeck构成，我们将它们放入一个列表中。
        for _ in range(num_layers):
            self.listLayers.append(BottleNeck(growth_rate=self.growth_rate, drop_rate=self.drop_rate))

    def call(self, x):
        for layer in self.listLayers.layers:
            x = layer(x)
        return x


# 过渡层 BN -> 1*1 Conv -> AvgPool  +SE 改进
class TransitionLayer(layers.Layer):
    # out_channels代表输出通道数
    def __init__(self, out_channels):
        super(TransitionLayer, self).__init__()
        self.bn = layers.BatchNormalization()  # BN
        self.conv = layers.Conv2D(filters=out_channels,  # 1*1 ConV
                                  kernel_size=(1, 1),
                                  strides=1,
                                  padding="same")
        self.pool = layers.AveragePooling2D(pool_size=(2, 2),  # 2倍下采样
                                            strides=2,
                                            padding="same")
        self.se_gap = layers.GlobalAveragePooling2D()
        self.se_avgp = layers.AveragePooling2D(pool_size=3, strides=3)
        # self.se_avgp = layers.MaxPooling2D(pool_size=3, strides=3)  # 最大池化
        self.se_resize = layers.Reshape((1, 1, out_channels))
        self.se_fc1 = layers.Dense(units=out_channels // 16, activation=tf.keras.activations.relu)
        self.se_fc2 = layers.Dense(units=out_channels, activation=tf.keras.activations.sigmoid)

    def call(self, inputs):
        x = self.bn(inputs)
        x = tf.keras.activations.relu(x)
        x = self.conv(x)
        x = self.pool(x)

        # 全局平均池化
        #         print(x)
        b = self.se_gap(x)
        #         print("GAP:",b)
        b = self.se_resize(b)

        # 局部平均池化
        c = x
        c = self.se_avgp(c)
        c = layers.Reshape((1, 1, c.shape[1] * c.shape[2] * c.shape[3]))(c)

        d = tf.concat([b, c], axis=-1)
        d = self.se_fc1(d)
        d = self.se_fc2(d)
        d = layers.Multiply()([x, d])
        return d


# DenseNet整体网络结构
class DenseNet(tf.keras.Model):
    # num_init_features:代表初始的通道数，即输入稠密块时的通道数
    # growth_rate:对应的是论文中的增长率k，指经过一个BottleNet输出的特征图的通道数
    # block_layers:每个稠密块中的BottleNet的个数
    # compression_rate:压缩因子，其值在(0,1]范围内
    # drop_rate：失活率

    def __init__(self, num_init_features, growth_rate, block_layers, compression_rate, drop_rate):
        super(DenseNet, self).__init__()
        # 第一层，7*7的卷积层，2倍下采样。
        self.conv = layers.Conv2D(filters=num_init_features,
                                  kernel_size=(7, 7),
                                  strides=2,
                                  padding="same")
        self.bn = layers.BatchNormalization()

        # 最大池化层，3*3卷积核，2倍下采样
        self.pool = layers.MaxPool2D(pool_size=(3, 3), strides=2, padding="same")

        # 稠密块 Dense Block(1)
        self.num_channels = num_init_features
        self.dense_block_1 = DenseBlock(num_layers=block_layers[0], growth_rate=growth_rate, drop_rate=drop_rate)
        # 该稠密块总的输出的通道数
        self.num_channels += growth_rate * block_layers[0]
        # 对特征图的通道数进行压缩
        self.num_channels = compression_rate * self.num_channels
        # 过渡层1，过渡层进行下采样
        self.transition_1 = TransitionLayer(out_channels=int(self.num_channels))

        # 稠密块 Dense Block(2)
        self.dense_block_2 = DenseBlock(num_layers=block_layers[1], growth_rate=growth_rate, drop_rate=drop_rate)
        self.num_channels += growth_rate * block_layers[1]
        self.num_channels = compression_rate * self.num_channels
        # 过渡层2，2倍下采样，输出：14*14
        self.transition_2 = TransitionLayer(out_channels=int(self.num_channels))

        # 稠密块 Dense Block(3)
        self.dense_block_3 = DenseBlock(num_layers=block_layers[2], growth_rate=growth_rate, drop_rate=drop_rate)
        self.num_channels += growth_rate * block_layers[2]
        self.num_channels = compression_rate * self.num_channels
        # 过渡层3，2倍下采样
        self.transition_3 = TransitionLayer(out_channels=int(self.num_channels))

        # 稠密块 Dense Block(4)
        self.dense_block_4 = DenseBlock(num_layers=block_layers[3], growth_rate=growth_rate, drop_rate=drop_rate)

        self.avgpool = layers.GlobalAveragePooling2D()

        self.fc = layers.Dense(units=num_class, activation=tf.keras.activations.softmax)

    def call(self, inputs):
        x = self.conv(inputs)

        #         print(x)
        x = self.bn(x)
        x = tf.keras.activations.relu(x)

        x = self.pool(x)
        x = self.dense_block_1(x)
        x = self.transition_1(x)
        x = self.dense_block_2(x)
        x = self.transition_2(x)
        x = self.dense_block_3(x)
        x = self.transition_3(x, )
        x = self.dense_block_4(x)

        x = self.avgpool(x)
        x = self.fc(x)

        return x

model = DenseNet(num_init_features=224, growth_rate=32, block_layers=[6, 12, 24, 16], compression_rate=0.5,
                 drop_rate=0.5)

model.build(input_shape=(None, 224, 224, 3))
####这个是原来没有注释summary，但是加注意力机制的话必须要进行注释########
model.summary()

model.compile(optimizer=optimizers.Adam(lr=0.0001),
              loss=tf.keras.losses.CategoricalCrossentropy(),
              metrics=['accuracy'])

history = model.fit(
    train_db,
    epochs=500,
    validation_data=val_db
)

plt.plot(history.history['accuracy'],color='#FF0000')
plt.plot(history.history['val_accuracy'],color='#008000')
# plt.title('SE-DenseNet-a')
plt.title('CERNet-accuracy')
plt.legend(['training', 'validation'], loc='lower right')
plt.xlabel("epoch")
plt.ylabel("accuracy")
plt.rcParams['savefig.dpi'] = 600  # 图片像素
plt.rcParams['figure.dpi'] = 600  # 分辨率
plt.savefig('./CERNet-accuracy acc_noaug.png')
plt.show()
test_loss, test_acc = model.evaluate(test_db)
print(test_acc)

# In[282]:


plt.plot(history.history['loss'],color='#FF0000')
plt.plot(history.history['val_loss'],color='#008000')
# plt.title('SE-DenseNet-a')
plt.title('CERNet-loss')
plt.legend(['training', 'validation'], loc='upper right')
plt.rcParams['savefig.dpi'] = 600  # 图片像素
plt.rcParams['figure.dpi'] = 600  # 分辨率
plt.savefig('./CERNet-loss loss_noaug.png')
plt.xlabel("epoch")
plt.ylabel("loss")
plt.show()
