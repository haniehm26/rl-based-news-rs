## News Recommender System

This is a News Recommender System implemented in Python, using an Online Reinforcement Learning approach, based on [MIND](https://www.microsoft.com/en-us/research/publication/mind-a-large-scale-dataset-for-news-recommendation/) dataset.

### Model Structure
In each iteration, user id must be submitted. Then, news title and news abstract will be presented. Next, the model needs user feedback.

<hr>
#### User ID
User id must be something like: U687515, U192112, U629430, U449564, U24161, U79744, U219005, ...


<input type="text" id="user-id" placeholder="User ID" value=""/>
<button type="submit" id="submit-id">submit</button>

<script>
  let user_id = document.getElementById("user-id").value;
  let api_url = `http://127.0.0.1:8000/recommend-news/${user_id}`;
  fetch(api_url)
        .then(
            (res) => {
                if (res.ok) {
                    return res.json();
                }
            }
        )
        .then(
            (res) => {
                document.getElementById("news-title-p").innerHTML = res.news.title;
                document.getElementById("news-abst-p").innerHTML = res.news.abstract;
            }
        )
</script>


#### News Title
<p id="news-title-p">blah blah blah</p>

#### News Abstract
<p id="news-abst-p">blah blah blah</p>

#### Read More?
<div>
  <button type="botton" id="yes">Yes</button>  
  <button type="botton" id="no">No</button>
</div>
<br>
<hr>

### Model Tracking
```markdown
Hi :)
Hi :)
Hi :)
```

### Contact
+ [Hanieh Mahdavi](https://www.linkedin.com/in/hanieh-mahdavi)
+ [Mohammadamin Farajzadeh](https://www.linkedin.com/in/mohammadamin-farajzadeh-bb050919a)
+ [Mahta Zarean](https://www.linkedin.com/in/mahta-zarean-9b7184198)
+ [Amir Asrzad](https://www.linkedin.com/in/amir-asrzad/)

### Acknowledgement
Special thanks to [Matin Zivdar](https://www.linkedin.com/in/matin-zivdar/) for his sincerely guidance and mentorship.
Many thanks to [Rahnema College](https://rahnemacollege.com/) for their fantastic internship program, [Azin Azarkar](https://www.linkedin.com/in/azin-azarkar-8829b6183), [Yasin Orouskhani](https://www.linkedin.com/in/yasinorouskhani/), and everyone else who helped us through this project.
