# Why you should drop out...
## ...or How youtube became everyone's favourite teacher

## Abstract
Learning is a fundamental process that began as a survival tool and evolved to encompass gaining knowledge, skills, and even reshaping our values. For most of history, knowledge was limited to what we could learn in person. The printing press expanded this, making information accessible worldwide. Later, the telegraph and telephone made real-time communication possible, allowing us to reach out instantly.

But sometimes, text isn't enough—enter YouTube. With videos available on virtually any topic, from Nobel Prize reactions to cooking tutorials, YouTube is a global classroom at our fingertips, available anytime, anywhere. In this project, we explore how YouTube has become our modern teacher, investigating trends and learning habits shaped by its educational content.

## Research Questions
#### What is YouTube’s teaching style?
- What categories of content are posted and appreciated?
- Do people prefer quick tutorials over lengthy explanations?

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
We will use [YouNiverse](https://zenodo.org/records/4650046) dataset, containing data about the content published on YouTube between May 2005 and October 2019. Since it is large in size, we have extracted video metadata in batches as shown in `data_extraction.ipynb` and decided to avoid using comment metadata, thus handling much smaller amounts of data.

The video metadata extracted corresponds to all videos listed as “Educational” by their creator: ~3.8M videos across ~25k channels. The category of the channel is not as relevant as it is determined by that of the 10 most recent videos. 

### Complementary dataset
We scraped the country of origin for approximately 25,000 YouTube channels using the YouTube Data API v3. Since information for deleted channels is no longer available through the API, we supplemented this with data scraped from both the YouTube API and the Web Archive (web.archive.org) version of Social Blade (socialblade.com).

### Data Enrichment
Using a LLM model [BART], we classify a sample of 50k educational videos into subcategories:
'Accademic', 'Edutenainment', 'Hobbies'.

## Methods
1. Extract data by batches → only select videos categorized as “Education”
2. Crawl for “country of origin” feature
3. Clean strings for title and tags of videos (keep ASCII characters, remove emojis and symbols) 
4. Manual keyword search to classify subcategories of content
4. BART classification for purpose labels (accademic, edutainment, hobby)
4. Using statistical tools to answer our questions
5. Plot information obtained in a comprehensive and interactive way

## Project Structure

The directory structure of our project looks like this:

```
├── data                        <- Project data files
│   ├── derivatives             <- produced data
│      ├── time_series_df           <- content creation over time for a selected topic
│   ├── figures
│      ├── exploration              <- exploratory plots of the data
│      ├── confusion matrices       <- files to plot confusion matrices
│      ├── cross_correlation_plots  <- cross correlation of different time series
│      ├── time_series_plots        <- Evolution of different topics' coverage over time put into perspective with different events in their fields
│      ├── causality                <- DWT plots and Granger causality heatmaps
│
├── pipelines                   <- useful pipelines developped to answer the research questions
│      ├── ...
│
├── results.ipynb              <- summary of the final status of our project
├── time_series.py             <- time series discretization function
├── topic_filtering.py         <- extract keywords from a wikipedia text to determine most frequent words in a topic
├── utils.py                   <- all sorts of handy functions
├── classify.py                <- functions to classify text with a given model
├── visualisation.py           <- plotting functions
├── config.py                  <- file containing global variables as well as label lists to feed LLMs
├── causality.py               <- causality tools
├── country_scraping.py        <- script to scrape country feature for channels using YouTube’s API and Webarchive of Social Blade
├── .gitignore                 <- List of files ignored by git
├── sickada_env.yml            <- File for installing python dependencies
└── README.md
```

## Conda environment setup
You can find our environment file ** sickadata_env.yml** and create a conda environment using the following commands in a terminal after navigating to the repository:

<pre><code> conda env create -f sickada_env.yml </code></pre>

Then you can activate the environment using:

<pre><code> conda activate sickada_env </code></pre>
