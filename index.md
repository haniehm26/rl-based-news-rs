## News Recommender System

This is a News Recommender System implemented in Python, using an Online Reinforcement Learning approach, based on [MIND](https://www.microsoft.com/en-us/research/publication/mind-a-large-scale-dataset-for-news-recommendation/) dataset.

### Model Structure
In each iteration, user id must be submitted. Then, news title and news abstract will be presented. Next, the model needs user feedback.

<hr>

#### User ID
User ID must be something like: U687515, U192112, U629430, U449564, U24161, U79744, U219005, ...

<input type="text" id="user-id-input" placeholder="User ID" value=""/>
<button type="submit" id="submit-id" onclick="submit_id()">submit</button>

#### News Title
<p id="news-title-p">blah blah blah</p>

#### News Abstract
<p id="news-abst-p">blah blah blah</p>

#### Read More?
<div>
  <button type="botton" id="yes" onclick="submit_yes()">Yes</button>  
  <button type="botton" id="no" onclick="submit_no()">No</button>
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

<script>
  function submit_id(event) {
    let user_id = document.getElementById("user-id-input").value;
    recommend_news(user_id);
    document.getElementById("news-title-p").innerHTML = user_id.value;
  }
  
  function submit_yes(event) {
    let response = 1;
    get_user_response(response);
    document.getElementById("news-title-p").innerHTML = response;
  }
  
  function submit_no(event) {
    let response = -1;
    get_user_response(response);
    document.getElementById("news-title-p").innerHTML = response;
  }
  
  function recommend_news(user_id) {
    let api_url = `https://185.220.224.95:8000/recommend-news/${user_id}`;
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
  }
  
  function get_user_response(user_response) {
    let api_url = `https://185.220.224.95:8000/response/${user_response}`;
    fetch(api_url)
        .then(
            (res) => {
                if (res.ok) {
                    return res.json();
                }
            }
        )
  }
</script>
