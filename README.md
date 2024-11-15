# Why you should drop out...
## ...or How youtube became everyone's favourite teacher

## Abstract
Learning is a fundamental process that began as a survival tool and evolved to encompass gaining knowledge, skills, and even reshaping our values. For most of history, knowledge was limited to what we could learn in person. The printing press expanded this, making information accessible worldwide. Later, the telegraph and telephone made real-time communication possible, allowing us to reach out instantly.

But sometimes, text isn't enough—enter YouTube. With videos available on virtually any topic, from Nobel Prize reactions to cooking tutorials, YouTube is a global classroom at our fingertips, available anytime, anywhere. In this project, we explore how YouTube has become our modern teacher, investigating trends and learning habits shaped by its educational content.


## Research Questions
#### What is YouTube’s teaching style?
- What purpose do videos have? Are they academic-based or for edutainment (education + entertainment)?
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

The video metadata extracted corresponds to all videos listed as “Educational” by their creator: ~3.8M videos across ~25k channels. The category of the channel is not as relevant as it is determined by that of the 10 most recent videos. 

### Complementary dataset
We scraped the country of origin of ~25k channels through the YouTube data API v3. (Since deleted channels do not store this information we might attempt to complement this with scraping of the web archive of [socialblade](https://socialblade.com/), the actual website having a paid API.)

### Data Enrichment
Using a LLM model, we classify educational videos into subcategories:
- **Purpose**: describes the modality of the video (e.g. academic course VS tutorial VS documentary VS ...)
- **Level**: describes the estimated level of the target audience (advanced VS intermediate VS beginner)
- **Content**: describes the content of the video (e.g. science VS history VS home repair VS ...)
Currently, we consider two models for the classification: [BART or BERT](https://medium.com/@reyhaneh.esmailbeigi/bert-gpt-and-bart-a-short-comparison-5d6a57175fca). Depending on model performance and computation speed on a sample of ~450k videos (~12% of the Education set), we will choose one of the models to extend the analysis to the whole set of interest (~3.8M videos). The two classification pipelines (one per model) are described in **NOTEBOOK NAME FROM SRC**.

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

### Timeline 

![image](https://github.com/user-attachments/assets/b501c1f5-88b5-4dd8-8b5c-149c08303777)



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

