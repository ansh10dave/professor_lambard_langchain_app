# professor_lambard_langchain_app
Professor Lambard is an educational application that leverages the Llama model 3.2 to facilitate interactive learning. Users can upload PDFs of textbooks, such as NCERT books, and engage in conversational dialogues to clarify concepts and enhance understanding.

### Description

Professor Lambard is an interactive educational app built with React Native (using Expo) designed to assist students in understanding complex topics across various chapters. Users can ask questions related to different chapters, and the app fetches relevant answers through an AI-powered backend model. This app offers an easy-to-use interface with a chat-like interaction for seamless communication.

The app features a chapter picker, where users can select a chapter, ask questions, and view answers directly from the app. The backend model provides accurate responses to questions based on the context of the selected chapter.

### Features

Chapter Selection: Users can choose from various chapters (e.g., Fluid Mechanics, Thermodynamics, Waves) to ask questions.

Interactive Chat Interface: Users can ask questions, and the app responds with answers from the backend model.

AI-Powered Q&A: Questions are sent to an AI backend which processes the queries and provides context-aware answers.

Loading Indicator: The app shows a loading screen while waiting for a response from the backend.

Dynamic Context Management: The app retains the context of previous answers and updates as new questions are asked.

### How to Use

Install Expo CLI: If you haven’t already installed Expo, run the following command
Clone the repository: Clone the repo to your local machine using the following command
git clone https://github.com/your-username/professor-lambard.git
Install Dependencies: Navigate to the project folder and install the necessary dependencies

Backend Setup:

Ensure you have the backend running locally or deployed to a server. The app sends requests to the backend (specified in the backendURL variable) to process questions and fetch answers.
The backend should expose an endpoint (/chat) to handle POST requests.

### Start the App: To start the Expo project and run the app on your device or emulator, use:

    expo start

    Select Chapter:
        Tap on the chapter selection button to open a modal with a list of available chapters (e.g., Fluid Mechanics, Thermodynamics, Waves).
        Choose the desired chapter to set the context for your questions.

    Ask a Question:
        After selecting a chapter, type a question in the input field and tap the "Go" button (arrow) to send the question.
        The app will send the question to the backend and display the response in the conversation window.

    View Answer:
        Once the backend processes the question, the app will show the answer in a message bubble in the chat interface.
        If the backend is processing the request, the app will display a loading message.

#### Some images 
<img src="https://github.com/user-attachments/assets/a7f80f2b-91f8-4aa5-a534-afa8070d31f5" width="250">
<img src="https://github.com/user-attachments/assets/39fdbb41-c49c-4bc7-8ec9-2843eab7667f" width="250">
<img src="https://github.com/user-attachments/assets/0b3ac3f3-3852-4b0f-a238-50900ca132a8" width="250">





### Tech Stack

    Frontend: React Native, Expo
    Backend: Python (Flask/FastAPI) or any backend framework capable of processing POST requests and generating responses using AI models
    Icons: AntDesign for icons

### Contributing

    Fork the repository
    Create your feature branch (git checkout -b feature-name)
    Commit your changes (git commit -am 'Add feature-name')
    Push to the branch (git push origin feature-name)
    Create a new Pull Request

For any questions, feel free to open an issue on GitHub or contact me directly on my email ansh10dave@gmail.com. 



