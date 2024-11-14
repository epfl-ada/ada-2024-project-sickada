# Why you should drop out
## or How youtube became everyone's favourite teacher

## Abstract
Learning is an essential cognitive process performed by all animals, originally to adapt and survive, now as a means to gain knowledge, build skills or even shift our attitudes and values. It’s about adding new tools to how we think, feel, and act. For millennia our knowledge was limited to the experience of the people we could directly interact with (aka [irl](https://www.reddit.com/r/AskOldPeople/comments/eaas63/what_was_life_like_before_the_internet/)). 
The invention of the press (thank you [Gutemberg and ancient Chinese people](https://www.history.com/topics/inventions/printing-press)) and the democratization of printed text changed this, by giving us access to books and news from everywhere. We also went a step further by enabling instantaneous communication through the telegraph and later the telephone. You could now press a few buttons to ask your dad for his [famous pie recipe](https://www.youtube.com/watch?v=MQCtLCxT1Xg) and have it instantly. With this, you could reach ([almost](https://www.youtube.com/watch?v=dGzCQUcC3Ac)) anyone in the world.

But what if your father sucked at [writing recipes](https://mikebakesnyc.com/how-to-develop-and-write-a-recipe/) or giving directions? A picture is worth a thousand words, and sometimes you need [24 of them per second](https://www.youtube.com/watch?v=dGzCQUcC3Ac). For minutes on end. YouTube allows people to share knowledge with you before you even have to ask for it (considered you don’t look before 2012), from [Nobel prize winner’s reaction to his achievement](https://www.youtube.com/watch?v=JhfDyBLRnrM), to a [day in the life of a neurosurgeon](https://www.youtube.com/watch?v=8RXEaAjpt4Q​​) or a [tutorial on how to make sushi with a professional chef](https://www.youtube.com/watch?v=nIoOv6lWYnk). The magic of it  also lies in its convenience; you can watch [whenever, wherever](https://www.youtube.com/watch?v=weRHyjj34ZE), at 10 am when you find the leak and need to understand [how you can fix this](https://www.youtube.com/watch?v=LTsgZo6VkkE) while maintaining pressure below your sink, or at 10 pm when you desperately try to make that sushi hold (seriously, [how do they make it seem so easy?](https://www.youtube.com/watch?v=dGzCQUcC3Ac)) 
You get the point, learning is now at the tip of your finger, and we consume tremendous amounts of it. In this project we will attempt to show you the evolution of our learning habits through our new teacher: YouTube. To do so we will investigate a few research questions to understand better the educational content landscape and the resulting consumption trends.


## Research Questions
#### What is YouTube’s teaching style?
- What purpose do videos have? Are they academic-based or for edutainment?
- What categories of content are posted and appreciated?
- What is the level of the target audience? Does this change for various categories?
- Do people prefer quick tutorials over lengthy explanations? Does this change between categories?

#### When did YouTube start teaching us?
- How has the volume of educational content changed over time? 
- Are there spikes in educational content upload and consumption over the years?

#### How does YouTube's curriculum change over the year?
- Are there specific times when educational content popularity spikes over the year?
- How does the information spread across categories in educational content?

#### How does YouTube adapt to different cultures?
- Is there a spatially dependent evolution in learning content’s volume, purpose or content?
- How does learning content evolve based on the location of creators?

## Dataset
We will use [YouNiverse](https://zenodo.org/records/4650046) dataset, containing data about the content published on YouTube between May 2005 and October 2019. Since it is large we have extracted video metadata in batches as shown in `data_extraction.ipynb` and decided to avoid using comment metadata, thus handling much smaller amounts of data.

The video metadata extracted corresponds to all videos listed as “Educational” by their creator. The category of the channel is not as relevant as it is determined by that of the 10 most recent videos. 

### Complementary dataset
We scraped the country of origin of ≈25k channels through the YouTube data API v3. (Since deleted channels do not store this information we might attempt to complement this with scraping of the web archive of [socialblade](https://socialblade.com/), the actual website having a paid API.)

### Data Enrichment
To obtain the following labels, we implemented two pipelines based on various LLMs as described in ??notebook names??. We intend to compare the models’ performances to decide which one to use going further.
**Purpose**: describing the type of educational content, e.g. lecture or academic course vs hacks.
**Level**: reflecting the estimated level of understanding of the target audience, e.g. advanced vs beginner.
**Content**: the subcategories of educational content, e.g. science or technology vs home repair or renovation.


## Methods
1. Extract data by batches → only select videos categorized as part of  “education”
2. Crawl for “country of origin” feature
3. Clean strings for title and tags of videos (keep ASCII characters, remove emojis and symbols) before feeding to BART or BERT (depending on which performs best across accuracy and time consumption) to assign additional labels.
4. Observe correlations between features to answer our questions.
5. Plot information obtained in a comprehensive and potentially interactive way


## Organization within the team

### A list of internal tasks for P2 milestone:
**Florian**: implementation of BART pipeline

**Frédéric**: implementation of BERT pipeline

**Gonçalo**: data scraping + BERT pipeline

**Viva**: data extraction + data exploration

**Yann**: implementation of BART pipeline + data exploration

## Timeline 

Add a Gantt chart?


## Project Structure

The directory structure of new project looks like this:

```
├── data                        <- Project data files
│
├── data_extraction.ipynb       <- extract data from video metadata file and save as multiple batches
├── country_scraping.ipynb      <- script to scrape country feature for channels using YouTube’s API
├── labelling_BERT.ipynb        <- script to label videos’ purpose, complexity and content using google’s BERT
├── labelling_BART.ipynb        <- script to label videos’ purpose, complexity and content using Meta’s BART
├── data_exploration.ipynb      <- script to produce figures in `/figures`
├── config.py                   <- file containing global variables as well as label lists to feed LLMs
│
├── figures
│   ├── exploration             <- Data directory
│
├── .gitignore                  <- List of files ignored by git
├── pip_requirements.txt        <- File for installing python dependencies
└── README.md
```


## Conda environment setup
You can find our environment file ** sickadata_env.yml** and create a conda environment using the following commands in a terminal after navigating to the repository:

<pre><code> conda env create -f sickada_env.yml </code></pre>

Then you can activate the environment using:

<pre><code> conda activate sickada_env </code></pre>

