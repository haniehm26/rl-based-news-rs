// set onclick
document.getElementById("submit-id").onclick = submit_id;
document.getElementById("submit-response").onclick = submit_response;

submit_id_btn = document.getElementById('submit-id')
submit_response_btn = document.getElementById('submit-response')

var disable_submit = true
submit_response_btn.disabled = disable_submit;

function submit_id(event) {
    submit_id_btn.disabled = !disable_submit;
    let user_id = document.getElementById("user-id").value;
    recommend_news(user_id)
    submit_id_btn.disabled = disable_submit;
    submit_response_btn.disabled = !disable_submit;
    // to stable the result and prevent refreshing the window too fast
    event.preventDefault();
}

function submit_response(event) {
    let yes = document.getElementById("yes");
    let no = document.getElementById("no");
    let response = 0
    if (yes.checked == true) {
        response = 1
    }
    else if (no.checked == true) {
        response = -1
    }
    get_user_response(response)
    submit_id_btn.disabled = !disable_submit;
    submit_response_btn.disabled = disable_submit;
    let element = document.getElementsByName("response");
    for (let i = 0; i < element.length; i++) {
        element[i].checked = false;
    }
    // to stable the result and prevent refreshing the window too fast
    event.preventDefault();
}

function recommend_news(user_id) {
    // let api_url = `https://185.220.224.94:8000/recommend-news/${user_id}`;
    let api_url = `http://185.220.224.72:8000/recommend-news/${user_id}`;
    // fetch url and make a get request
    fetch(api_url)
        .then(
            (res) => {
                // if request is okay return its json, otherwise display an error
                if (res.ok) {
                    return res.json();
                }
            }
        )
        .then(
            (res) => {
                // using two values of the returned 
                document.getElementById("news-title-p").innerHTML = res.news.title;
                document.getElementById("news-abst-p").innerHTML = res.news.abstract;
            }
        )
}

function get_user_response(user_response) {
    // let api_url = `https://185.220.224.94:8000/response/${user_response}`;
    let api_url = `http://185.220.224.72:8000/response/${user_response}`;
    // fetch url and make a get request
    fetch(api_url)
        .then(
            (res) => {
                // if request is okay return its json, otherwise display an error
                if (res.ok) {
                    return res.json();
                }
            }
        )
}