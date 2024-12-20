DRIVE_PATH_VIVA = r"/Volumes/VIVA HD/ADA" 
DRIVE_PATH_FLORIAN = r"C:\Users\Flori\Docs\Python\M2_S3_ADA\Project - data\Education_videos_5.csv" # for a single sample for now

content_categories = {  
    '20': "history", # wwi wwii
    '21': "religion or spirituality",
    '22': "philosophy or ethics",
    '24': 'geopolitics',
    '25': "cryptocurrency",
    '29': "history random", # litteraly the word 'history' # to be deleted

    '3':"food or cooking",
    '39': 'food random', # litteraly the word 'cooking'

    '4': 'programming',
    '41': 'machine learning',
    '49': 'programming random',

    '5': 'children content',

    '6': 'edutainment', # litteraly the word
    '61': 'science and tech',
    '62': "wildlife or animals or nature",
    '63': "foreign language or language proficiency",
    '65': 'home repair or renovation',
    '69': 'engineer',

    '7' : "sports",
    '71': 'football',
    '72': 'basketball',
    '73': 'american football',
    '74': 'cricket',
    '75': 'baseball',
    '79': 'sports random',

    '8': "music",
    '81': 'piano tutorial',
    '82': 'guitar tutorial',
    '83': 'violin tutorial',
    '84': 'drums tutorial',
    '85': 'ukulele tutorial',
    '86': 'artists 2010',
    '87': 'classical music',
    '88': 'dances',
    '89': 'music random',

    '90' : 'chess',

    '9' : 'gaming',
    '91': 'roblox',
    '92' : 'minecraft',
    '94': 'pubg',
    '95' : 'league of legends',
    '96': 'call of duty',
    '97': 'super mario',
    '98': 'pokemon',
    '99': 'gaming random', # 'gaming'

    'a': 'audiobooks',
    'android' : 'android', # to sort
    'q': 'conspiracy',
    's': 'spam', # not the ham of course
    'unclass': 'unclassified',
    'life': 'lifestyle'

    #'81': 'puzzles', # 5:  logic & riddles
    #'7': "photography or videography or filmaking",
    # 14: "health or medicine",
    # 15: "travel",
    # 16: "motivational or personal development",
    # 18: "beauty or fashion",
    # 22: "psychology",
    # 23: "climate or environment",
}

label_clustering = {
    'Science': ['61', '69'],
    'Programming': ['4', '41'],
    'Sport': ['7', '71', '72', '73', '74', '75'],
    'Edutainment': ['24', '20', '21', '22', '25'],
    'Gaming': ['91','92', '94', '95', '96', '97', '98', '9'],
    'Music': ['81', '82', '83', '84', '85', '87', '88', '8'],
    'Cooking': ['3'],
    'Home repair': ['65'],
    'Wildlife': ['62'],
    'Language learning': ['63']
}

 