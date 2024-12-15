import os.path as op
import pathlib

DRIVE_PATH_VIVA = r"/Volumes/VIVA HD/ADA" 
DRIVE_PATH_FLORIAN = r"C:\Users\Flori\Docs\Python\M2_S3_ADA\Project - data\Education_videos_5.csv"
config_path = pathlib.Path(__file__).parent.resolve() # path of the repository

path_data = op.join(config_path, 'data', 'raw')
path_deriv = op.join(config_path, 'data', 'derivatives')
path_figures = op.join(config_path, 'data', 'figures')
path_metadata = op.join(path_data, "yt_metadata_en.jsonl.gz")
path_channels = op.join(path_data, "df_channels_en.tsv.gz")
path_edu = op.join(path_deriv, "Education_videos_{}.csv")

N_BATCHES = 8

purpose_labels = [
    "lecture or academic course", #exercise
    #"study-tips or test preparation",
    "hacks", 
    "conference",
    "tutorial or DIY",
    "interview or Q&A or review", #FIND BETTER
    "kids content",
    "entertaining explanation or science popularization",
    "documentary" #research based
]

level_labels = [
    "beginner",
    "intermediate",
    "advanced",
]

content_labels = [
    "science or technology",
    "music or art",
    "photography or videography or filmaking",
    "gaming",
    "chess or puzzles or logic", #riddles
    "religion or spirituality",
    "phylosophy or ethics",
    "history or politics",
    "economics or business",
    "financial education",
    "cryptocurrency",
    "food or cooking",
    "sport",
    "health or medicine",
    "travel",
    "motivational or personal development",
    "home repair or renovation",  
    "beauty or fashion",
    "programming tools or coding",
    "foreign language or language proficiency",
    "sociology or culture",
    "psychology",
    "climate or environment",
    "wildlife or animals or nature" #segment?
]

# those that end with 9 are the ones where a general keyword was used
content_categories = {  
    '20': "history",
    '21': "religion or spirituality",
    #7: "phylosophy or ethics",
    #9: "economics or business",
    #10: "financial education",
    '25': "cryptocurrency",

    '4': 'programming', # needs to come before gaming
    '41': 'machine learning',
    '49': 'programming random',

    '5': 'children content',
    '6': 'edutainment', # litteraly that word
    #'61' : 'science and tech',
    '64':"food or cooking", 
    '65': 'home repair or renovation',

    '8' : 'chess',
    #'81': 'puzzles', # 5:  logic & riddles

    '9' : 'gaming',
    '91': 'roblox',
    '92' : 'minecraft',
    '93': 'fortnite',
    '94': 'pubg',
    '95' : 'league of legends',
    '96': 'call of duty',
    '99': 'gaming random',

    '7' : "sports",
    '71': 'football',
    '72': 'basketball',
    '73': 'american football',
    '74' : 'cricket',
    '75': 'baseball',
    '79': 'sports random',

    '8': "music",
    '81': 'piano tutorial',
    '82': 'guitar tutorial',
    '83': 'violin tutorial',
    '84': 'drums tutorial',
    '85': 'ukulele tutorial',
    '86': 'artists 2010', #vocal tutorial
    '87': 'classical music',
    '89': 'music random',

    'x': 'audiobooks',
    'android' : 'android', # to sort
    'q': 'conspiracy',

    #'': "wildlife or animals or nature", 
    #'7': "photography or videography or filmaking",
    # 12: 
    # 14: "health or medicine",
    # 15: "travel",
    # 16: "motivational or personal development",
     
    # 18: "beauty or fashion",
    # 20: "foreign language or language proficiency",
    # 21: "sociology or culture",
    # 22: "psychology",
    # 23: "climate or environment",
}

inverted_categories ={
    'programming': '4',
    'machine learning': '41',
    'gaming': '9',
    'roblox': '91',
    'minecraft': '92',
    'fortnite': '93',
    'pubg': '94',
    'league of legends': '95',
    'call of duty': '96',

    'music': '8',
    'piano tutorial': '81',
    'guitar tutorial': '82',
    'violin tutorial': '83',
    'drums tutorial': '84',
    'ukulele tutorial': '85'
}
