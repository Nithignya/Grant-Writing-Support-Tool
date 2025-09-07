import React, { useState } from "react";
import InputForm from "./InputForm";
import OutputDisplay from "./OutputDisplay";
import VoiceInput from "./VoiceInput";

function Body() {
  const [aiOutput, setAiOutput] = useState(""); 
  const [isLoading, setIsLoading] = useState(false); 
  const [transcription, setTranscription] = useState("");

  const handleFormSubmit = async (data) => {
    setIsLoading(true); // Show loading indicator
    setAiOutput(""); // Clear previous AI output

    try {
        const API_URL = "https://grant-writing-backend.onrender.com";
        const response = await fetch(`${API_URL}/grant_writer/generate/`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ ...data, transcription }),
      });
      console.log({"Submitting data to backend": data, transcription}); // data which is being sent to the backend

      if (!response.ok) {
        throw new Error("Failed to fetch AI response.");
      }

      const result = await response.json();
      console.log("Received AI response from backend:", result);
      setAiOutput(result.response); // Update with AI response
    } catch (error) {
      console.error("Error while fetching AI response:", error);
    } finally {
      setIsLoading(false); // Hide loading indicator
    }
  };

  return (
    <section className="flex justify-center my-8">
      <div className="w-3/12  p-4">
        <InputForm onSubmit={handleFormSubmit} />
        <VoiceInput onTranscription={(text) => setTranscription(text)} />
      </div>

      <div className="w-7/12 p-4">
        <OutputDisplay aiOutput={aiOutput} isLoading={isLoading} />
      </div>
    </section>
  );
}

export default Body;
