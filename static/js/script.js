
var startButton = document.querySelector("#startQuiz");     //Main page start button
var timer = document.querySelector("#timer");   //Timer when quiz starts
var mainContent = document.querySelector("#mainContent");   //Start page content div
var questionEl = document.querySelector("#title");  //card title
var quizContent = document.querySelector("#quizContent");   //Question options div
var resultDiv = document.querySelector("#answer");  //Div for showing answer is correct/wrong
var completeTest = document.querySelector("#completeTest");    //Div for Displying final scores when quiz completed
var highscoresDiv = document.querySelector("#highscores");  //Div for showing highscores
var navhighscorelink = document.querySelector("#navhighscorelink");     //highscore navigation link
var navlink = document.getElementById("navhighscorelink");

var secondsLeft = 300, questionIndex = 0,correct = 0 ;
var totalQuestions = questions.length;
var question , option1, option2, option3 ,option4 ,ans, previousScores;
var choiceArray = [], divArray = [];

//create buttons for choices
for(var i = 0 ; i < 4 ; i++){
    var dv = document.createElement("div");
    var ch = document.createElement("button");
    ch.setAttribute("data-index",i);
    ch.setAttribute("class","btn rounded-pill mb-2");
    ch.setAttribute("style","background:#5f9ea0");
    choiceArray.push(ch);
    divArray.push(dv);
}

//Start Quiz function
function startQuiz(){

    startTimer();
    buildQuestion();

}

//function to start timer when quiz starts
function startTimer(){

    var timeInterval = setInterval(function(){

        secondsLeft--;

        timer.textContent = "Time : "+ secondsLeft+ " sec";
//        if(secondsLeft <= 60){
//            timer.textContent = "Time : 1 min";
//        }

        if(secondsLeft <= 0 || (questionIndex > totalQuestions-1)){

            resultDiv.style.display = "none";
            quizContent.style.display = "none";
            viewResult();
            clearInterval(timeInterval);
            timer.textContent = "";
        }

    },1000);
}


function buildQuestion(){

    //hides start page content
    questionEl.style.display= "none";
    mainContent.style.display = "none";
    quizContent.style.display= "none";

    if(questionIndex > totalQuestions - 1){
        return;
    }
    else{
        ans =  questions[questionIndex].answer;

        //Display Question
        questionEl.innerHTML = questions[questionIndex].title;
        questionEl.setAttribute("class","text-left");
        questionEl.style.display= "block";

        for(var j = 0 ; j < 4 ; j++){
            var index = choiceArray[j].getAttribute("data-index");
            choiceArray[j].textContent = (+index+1) +". "+questions[questionIndex].choices[index];
            divArray[j].appendChild(choiceArray[j]);
            quizContent.appendChild(divArray[j]);
        }

    }
    quizContent.style.display= "block"; // Display options
}

// Event Listener for options buttons
quizContent.addEventListener("click",function(event){

    var element = event.target;
    var userAnswer = element.textContent;
    var userOption   = userAnswer.substring(3, userAnswer.length);

        if(userOption === ans){
            correct++;

            resultDiv.style.display = "block";
        }
        else {
            secondsLeft -= 10;

            setTimeout(function(){
                resultDiv.textContent = "";
            },500);
        }

        questionIndex++;
        buildQuestion();
});


//Function to show score when quiz completes
function viewResult(){

    questionEl.innerHTML = "Great Job! Your Test Completed!";
    questionEl.style.display= "block";


    var scoreButton = document.createElement("button");     //Submit User score
    scoreButton.setAttribute("class","btn rounded-pill mb-2 ml-3 mt-2");
    scoreButton.setAttribute("style","background:#5f9ea0");
    scoreButton.textContent = "Submit";
    completeTest.appendChild(scoreButton);

    scoreButton.addEventListener("click",function(){
    jQuery.noConflict();
    jQuery(document).ready(function($) {
    var inputData = correct;
                $.ajax({
                    type: "POST",
                    url: "/exam",
                    contentType: "application/json",
                    data: JSON.stringify({input: inputData}),
                    success: function(response) {
                        console.log(response);
                        console.log(response['output']);

                        // Redirect to result.html on success
                        window.location.href = "/"+response['link']+"/".concat(response['output']);
                    },
                    error: function(xhr, status, error) {
                        // Handle errors here if needed
                    }

                });
    });
    });
}
/*
//Function to store highscores
function storeScores(event){

    event.preventDefault();
    var userName = document.querySelector("#nameInput").value.trim();

    if(userName === null || userName === '') {
        alert("Please enter user name");
        return;
     }

      //Create user object for storing highscore
        var user = {
            name : userName,
            score : correct
        }

        console.log(user);

        previousScores = JSON.parse(localStorage.getItem("previousScores"));    //get User highscores array in localStorage if exists

        if(previousScores){
            previousScores.push(user); //Push new user scores in array in localStorage
        }
        else{
            previousScores = [user];    //If No user scores stored in localStorage, create array to store user object
        }

        // set new submission
        localStorage.setItem("previousScores",JSON.stringify(previousScores));

        showHighScores(); // Called function to display highscores

}
*/
//Start button event listener on start page which starts quiz
$(document).ready(function() {
    startButton.addEventListener("click",function(){
    jQuery.noConflict();
    jQuery(document).ready(function($) {
                $.ajax({
                    type: "POST",
                    url: "/exam",
                    contentType: "application/json",
                    data: JSON.stringify({input:''}),
                    success: function(response) {
                        startQuiz();
                    },
                    error: function(xhr, status, error) {
                        // Handle errors here if needed
                    }

                });
    });
    });
});



