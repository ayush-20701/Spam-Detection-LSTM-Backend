import { useState, useEffect } from "react";
import "./styles/app.css"
import "./styles/responsive.css"
import { DotLottieReact } from '@lottiefiles/dotlottie-react';
import ham from "./assets/hamburger.svg"
import 'animate.css'

function App() {
  const [message, setMessage] = useState("");
  const [result, setResult] = useState("");
  const [loading, setloading] = useState(false)
  const [showResult, setShowResult] = useState(false)

  const handleSubmit = async (e) => {
    e.preventDefault();
    const cleanedMessage = message.trim(); // Remove spaces & newlines
    try {
      setResult("")
      setloading(true)
      setShowResult(false)
      // Call your Flask API
      const response = await fetch("http://127.0.0.1:5000/predict", {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify({ message: cleanedMessage }),
      });
      // console.log(cleanedMessage);
      const data = await response.json();
      setTimeout(() => {
        setResult(data.prediction === "Spam" ? "Beware! It's a spam." : "Good to go! It's spam free.")
        setShowResult(true)
        setloading(false)
      }, 1500);

    } catch (error) {
      console.error("Error:", error);
      setResult("Error making request");
      setShowResult(true)
    }
  };

  // Detect Enter key in textarea
  const handleKeyDown = (e) => {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault(); // Prevent new line in textarea
      const button = document.querySelector("button[type='submit']");
      if (button) {
        button.classList.add("pressed"); // Add effect
        setTimeout(() => button.classList.remove("pressed"), 150); // Remove effect after 150ms
      }
      handleSubmit(e); // Submit the form
    }
  };

  //add effect on enter button clicked
  const handleButtonPress = (e) => {
    const button = e.target;
    button.classList.add("pressed");
    setTimeout(() => button.classList.remove("pressed"), 150);
  };

  return (
    <div className="main">

      <div className="alert alert-warning alert-dismissible fade show" role="alert">
        The backend Flask API is not deployed yet. The ML model returns result only on localhost currently. So, result would not be displayed. Sorry for the inconvenience!
        <button type="button" className="btn-close" data-bs-dismiss="alert" aria-label="Close"></button>
      </div>
      <DotLottieReact className="bg-animation" src="https://lottie.host/4bb0b478-0598-4143-b847-9cbb1eb03e89/w3gRuWcuNk.lottie" loop autoplay />
      {/* <DotLottieReact src="https://lottie.host/01508e0a-3ef4-4230-8e2c-740a3b8cb52d/ytGPDNMc0L.lottie" loop autoplay /> */}
      <div className="navbar">
        <button>
          <img src={ham} alt="ham icon" />
        </button>
        <div className="title">SMS Spam Detector</div>
      </div>
      <div className="desc">Enter a message below and let my ML model decide whether it's a spam</div>
      <form onSubmit={handleSubmit}>
        <textarea
          value={message}
          onChange={(e) => setMessage(e.target.value)}
          onKeyDown={handleKeyDown} // Call function when key is pressed
          placeholder="Enter your message here..."
          required
        />
        <button onMouseDown={handleButtonPress} onKeyDown={handleButtonPress} type="submit">Submit</button>
      </form>
      {loading && <DotLottieReact className="loading"
        src="https://lottie.host/01ae8253-0eb0-4fa3-97c0-ab55b5d592d4/Vn6yc09IPY.lottie"
        loop
        autoplay
      />}
      {result && showResult && <div className="res animate__animated animate__backInUp">
        {result}
      </div>}
      <div className="footer">&copy;This model and UI is developed by <a href="https://linktr.ee/ayush20701">Ayush Kumar</a></div>
    </div>
  );
}

export default App;