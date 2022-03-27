## Welcome to Recbrain Recommender System

This is a News Recommender System implemented in Python, using an Online Reinforcement Learning approach, based on [MIND](https://www.microsoft.com/en-us/research/publication/mind-a-large-scale-dataset-for-news-recommendation/) dataset.

In each iteration, user id must be submitted, then news title and news abstract will be presented. Next, the model needs user feedback.

User ids must be something like: U687515, U192112, U629430, U449564, U24161, U79744, U219005, ...

### Model Structure
<h4>User ID</h4>
<input type="text" id="user-id" placeholder="User ID" value=""/>
<button type="submit" id="submit-id">submit</button>
<div>
  <h4 id="news-title">News Title</h4>
  <p id="news-title-p">blah blah blah</p>
  <h4 id="news-abst">News Abstract</h4>
  <p id="news-abst-p">blah blah blah</p>
</div>
<div>
<h4 id="user-response">Read More?</h4>
  <button type="botton" id="yes">Yes</button>  
  <button type="botton" id="no">No</button>
</div>
<br>

Markdown is a lightweight and easy-to-use syntax for styling your writing. It includes conventions for

```markdown
Syntax highlighted code block

# Header 1
## Header 2
### Header 3

- Bulleted
- List

1. Numbered
2. List

**Bold** and _Italic_ and `Code` text

[Link](url) and ![Image](https://github.com/haniehm26/rl-based-news-rs/blob/master/images/logo.png)
```

For more details see [Basic writing and formatting syntax](https://docs.github.com/en/github/writing-on-github/getting-started-with-writing-and-formatting-on-github/basic-writing-and-formatting-syntax).

### Jekyll Themes

Your Pages site will use the layout and styles from the Jekyll theme you have selected in your [repository settings](https://github.com/haniehm26/rl-based-news-rs/settings/pages). The name of this theme is saved in the Jekyll `_config.yml` configuration file.

### Support or Contact

Having trouble with Pages? Check out our [documentation](https://docs.github.com/categories/github-pages-basics/) or [contact support](https://support.github.com/contact) and we’ll help you sort it out.
