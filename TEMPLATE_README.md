# “ODD” ASPECTS OF AIRBNB

**Author**: Ben McCarty

---
# Overview

A one-paragraph overview of the project, including the business problem, data, methods, results and recommendations.

>Problem: Airbnb hosts are required to maintain a guest satisfaction score of 4+/5. When the What aspects of a host property are the strongest predictors of a guest review being a 4 or greater (out of 5)?

>Data: Airbnb data on hosts, properties, and guest reviews (scores and verbatims) for the Washington, DC market. Sourced from [**Inside Airbnb**](http://insideairbnb.com/get-the-data.html#:~:text=Washington%2C%20D.C.%2C%20District%20of%20Columbia%2C%20United%20States)

>Methods: Classification modeling to determine the probability of a 4+/5 rating for a given property record.

# ❌ Results: ???

# ❌ Recommendations: ???

---
# Business Problem

Summary of the business problem you are trying to solve, and the data questions that you plan to answer in order to solve them.


Questions to consider:

>* *What are the business's pain points related to this project?*
>   * Poor guest satisfaction negatively impacts revenue
>       * Guests less likely to return to the property/other properties within the brand
>       * Dissatisfied guests more likely to tell others about their bad experiences, decreasing the likelihood of new guests trying an Airbnb
>
>* *How did you pick the data analysis question(s) that you did?*
>   * Based on my experience in hospitality, I know that the guest's overall score is one of the top performance metrics, if not the most important
>       * When I was a hotel manager, I was curious about what aspects of the guest experience had the strongest impact on their review scores.
>   * The overall review scores for this dataset showed a large percentage of scores over 4.
>       * I wanted to know what would be the strongest predictors of a property falling below a score of 4.
>
>
>* *Why are these questions important from a business perspective?*
>    * Airbnb needs to be able to identify properties at risk of falling below a 4 rating.
>        * My project's results give insight into what features are strongest predictors of the sub-standard scores.
>       * The results can also give hosts priority items on which to focus to prevent and/or address low scores.
***

# Data

Describe the data being used for this project.


Questions to consider:
* *Where did the data come from, and how do they relate to the data analysis questions?*
    * Sourced from [*Inside Airbnb*](http://insideairbnb.com/index.html), *"an independent, non-commercial set of tools and data that allows you to explore how Airbnb is really being used in cities around the world."* [Source](http://insideairbnb.com/about.html#:~:text=Inside%20Airbnb%20is%20an%20independent%2C)
    * "The data utilizes public information compiled from the Airbnb web-site including the availabiity calendar for 365 days in the future, and the reviews for each listing." [Source](http://insideairbnb.com/about.html#:~:text=The%20data%20utilizes%20public%20information)
* *What do the data represent? Who is in the sample and what variables are included? What are their properties?*
    * Location: Washington, DC and nearby neighborhoods
    * Host details:
        * Number of listings per host
        * Whether/not host has a profile picture
        * Host verification methods (incl. via email, google, government ID, manual verification in-person and online)
    * Property details:
        * Number of beds, baths
        * Min, max nights
        * Availability in different time ranges (30, 60, 90, and 365 day windows)
        * Location (Lat/Long and neighborhood)
        * Price per night
* *What is the target variable?*
    * Predicting "review_scores_rating," the overall score for a host property

***

# Methods

Describe the process for analyzing or modeling the data.

Questions to consider:
* *How did you prepare, analyze or model the data?*
    * Preparation included:
        * Ignoring specific features of the dataset that were missing 25% or more data points
        * Ignoring entries that were missing entries for 6 or more of the total features
        * Converting data into more usable formats
        * Identifying the number of years of hosting experience
* *Why is this approach appropriate given the data and the business problem?*
    * Dropping features and properties with large amounts of missing values leads to more reliably accurate results than filling in the missing values based on the existing data
        * Filling in the data may result in poorer results
    * Raw data from source is provided in unusable formats
    * Being able to generate the data for the number of years of experience adds more insight to the model, increasing the model's performance.
***

## Results

Present your key results.

***
Questions to consider:
* How do you interpret the results?
* How confident are you that your results would generalize beyond the data you have?
***

Here is an example of how to embed images from your sub-folder:

### Visual 1
![graph1](./images/viz1.png)

## Conclusions

Provide your conclusions about the work you've done, including any limitations or next steps.

***
Questions to consider:
* What would you recommend the business do as a result of this work?
* What are some reasons why your analysis might not fully solve the business problem?
* What else could you do in the future to improve this project?
***

## For More Information

Please review our full analysis in [our Jupyter Notebook](./dsc-phase1-project-template.ipynb) or our [presentation](./DS_Project_Presentation.pdf).

For any additional questions, please contact **name & email, name & email**

## Repository Structure

Describe the structure of your repository and its contents, for example:

```
├── README.md                           <- The top-level README for reviewers of this project
├── dsc-phase1-project-template.ipynb   <- Narrative documentation of analysis in Jupyter notebook
├── DS_Project_Presentation.pdf         <- PDF version of project presentation
├── data                                <- Both sourced externally and generated from code
└── images                              <- Both sourced externally and generated from code
```
