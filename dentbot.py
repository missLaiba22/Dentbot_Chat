import os
from ibm_watsonx_ai.foundation_models import Model
import gradio as gr

# Function to get credentials from environment variables
def get_credentials():
    api_key = os.getenv("IBM_API_KEY")
    if not api_key:
        raise ValueError("API key is not set in environment variables.")
    return {
        "url": "https://us-south.ml.cloud.ibm.com",
        "apikey": api_key
    }

# Initialize Model
def initialize_model():
    model_id = "ibm/granite-13b-chat-v2"
    parameters = {
        "decoding_method": "greedy",
        "max_new_tokens": 1000,
        "stop_sequences": ["\n"],
        "repetition_penalty": 1
    }
    project_id = os.getenv("PROJECT_ID")

    if not project_id:
        raise ValueError("Project ID is not set in environment variables.")

    return Model(
        model_id=model_id,
        params=parameters,
        credentials=get_credentials(),
        project_id=project_id,
    )

# Initialize the model
model = initialize_model()

# Detailed context
context = """Help users schedule their dental appointments by collecting necessary information and checking availability from given context:
Dental Clinic Context
Clinic Name: Bright Smiles Dental Clinic

Address: 123 Health Street, Wellness City, HC 12345

Phone Number: +1-800-555-1234

Email: contact@brightsmilesdental.com

Website: www.brightsmilesdental.com

Operating Hours:

Monday to Friday: 9:00 AM - 6:00 PM
Saturday: 10:00 AM - 3:00 PM
Sunday: Closed
Available Treatments:

Routine Check-ups: Comprehensive examination of teeth and gums, including X-rays if needed.
Cleanings: Professional dental cleaning, including scaling and polishing.
Fillings: Treatment for cavities, including composite and amalgam fillings.
Extractions: Removal of decayed or damaged teeth.
Root Canals: Treatment for infected or damaged tooth roots.
Crowns and Bridges: Restoration of damaged or missing teeth with crowns and bridges.
Teeth Whitening: Cosmetic whitening procedures to enhance smile brightness.
Orthodontics: Braces and clear aligners for teeth straightening.
Periodontal Treatment: Treatment for gum disease and related conditions.
Emergency Care: Immediate care for dental emergencies such as severe pain, trauma, or infections.
Doctors Present:

Dr. Alice Johnson:

Specialization: General Dentistry
Experience: 15 years
Languages: English, Spanish
Dr. Michael Smith:

Specialization: Orthodontics
Experience: 10 years
Languages: English
Dr. Emily Brown:

Specialization: Periodontics
Experience: 12 years
Languages: English, French
Dr. David Lee:

Specialization: Endodontics (Root Canals)
Experience: 8 years
Languages: English
Appointment Scheduling:

Booking Methods: Appointments can be booked via phone, email, or the online booking system on the clinic’s website.
Cancellation Policy: Please cancel or reschedule at least 24 hours in advance to avoid a cancellation fee.
Additional Information:

Parking: Free parking available in the clinic's parking lot.
Insurance: Accepts most major dental insurance plans. Check with the clinic for specific coverage details.
Payment Options: Credit/Debit cards, cash, and financing options available.
Contextual Setup:
Clinic Timing and Availability Example:

Monday: 9:00 AM - 5:00 PM
Tuesday: 9:00 AM - 5:00 PM
Wednesday: Closed
Thursday: 9:00 AM - 6:00 PM
Friday: 9:00 AM - 5:00 PM
Saturday: 10:00 AM - 2:00 PM
Sunday: Closed
Doctors Available:

Dr. Smith: General Dentistry, available Monday and Tuesday afternoons.
Dr. Jones: Cosmetic Dentistry, available Monday afternoons and Tuesday mornings.
Dr. Lee: Orthodontics, available Tuesday afternoons.
Treatments Available:

Teeth Whitening: Available on Monday and Tuesday afternoons.

Input: Can you tell me about the available treatments at Bright Smiles Dental Clinic?
Output: At Bright Smiles Dental Clinic, we offer a range of treatments including:
Routine Check-ups
Cleanings
Fillings
Extractions
Root Canals
Crowns and Bridges
Teeth Whitening
Orthodontics
Periodontal Treatment
Emergency Care
For more details about any specific treatment, feel free to ask!

Input: What are the operating hours of Bright Smiles Dental Clinic?
Output: Bright Smiles Dental Clinic is open during the following hours:
Monday to Friday: 9:00 AM - 6:00 PM
Saturday: 10:00 AM - 3:00 PM
Sunday: Closed
If you need to book an appointment, you can do so during these hours!

Input: Who are the doctors available at the clinic?


Output: We have the following doctors at Bright Smiles Dental Clinic:
Dr. Alice Johnson - General Dentistry (15 years of experience)
Dr. Michael Smith - Orthodontics (10 years of experience)
Dr. Emily Brown - Periodontics (12 years of experience)
Dr. David Lee - Endodontics (8 years of experience)
You can book an appointment with any of them based on your needs!

Input: How can I schedule an appointment at Bright Smiles Dental Clinic?


Output: You can schedule an appointment at Bright Smiles Dental Clinic in the following ways:
Phone: Call us at +1-800-555-1234.
Email: Send your appointment request to contact@brightsmilesdental.com.
Online: Use our online booking system on our website www.brightsmilesdental.com.
If you need to cancel or reschedule, please do so at least 24 hours in advance to avoid any cancellation fees."

Input: What are the payment options available at the clinic?


Output: At Bright Smiles Dental Clinic, we accept the following payment options:
Credit/Debit cards
Cash
Financing options
We also accept most major dental insurance plans. Please check with us for specific coverage details.

Input: My preferred time is afternoon on Monday or Tuesday, is it possible?
Output: Yes, we have available slots for teeth whitening treatments on both Monday and Tuesday afternoons. You can see the following doctors:Dr. Smith is available on Monday and Tuesday afternoons.
Dr. Lee is available on Tuesday afternoons.
Please let me know which day you prefer and if you have any specific time in mind. I will then proceed to book your appointment.

Input: Can I get an appointment for a dental filling on Friday?
Output: Yes, we have available slots for dental fillings on Friday. The available times are:10:00 AM
12:00 PM
2:00 PM
4:00 PM Please choose a time that works best for you, and let me know if you have any specific preferences or concerns.

Input: What are the available times for an orthodontic consultation this week?
Output: We have several slots available for orthodontic consultations this week. The available times are:Monday: 3:00 PM, 4:00 PM
Wednesday: 10:00 AM, 2:00 PM
Friday: 11:00 AM, 1:00 PM Please let me know your preferred day and time, and if you have any specific questions or needs for the consultation.

Input: Can I schedule a dental cleaning appointment for next Tuesday morning?
Output: We have available slots for dental cleaning next Tuesday morning. The available times are:9:00 AM,10:00 AM, 11:00 AM Please let me know which time suits you, and if you have any preferences for a specific hygienist.

Input: Hi, I’d like to book a dental appointment. Can you help me with that?
Output: Hello! I’d be happy to help you schedule a dental appointment. What type of treatment are you interested in, and do you have a preferred date and time?"""


# Custom CSS for Gradio Interface
custom_css = """
.gradio-container {
    font-family: 'Arial', sans-serif;
    background-color: #e9ecef;
    border-radius: 15px;
    padding: 20px;
    max-width: 800px;
    margin: 0 auto;
    box-shadow: 0 4px 8px rgba(0, 0, 0, 0.1);
}
.gradio-title {
    font-size: 28px;
    font-weight: 700;
    color: #343a40;
    text-align: center;
    margin-bottom: 20px;
}
.gradio-description {
    font-size: 18px;
    color: #495057;
    text-align: center;
    margin-bottom: 30px;
}
.gradio-chatbot {
    background: linear-gradient(to bottom, #ffffff 0%, #f8f9fa 100%);
    border: 1px solid #dee2e6;
    border-radius: 15px;
    padding: 20px;
    box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
}
.gradio-input {
    border: 1px solid #ced4da;
    border-radius: 15px;
    padding: 12px;
    font-size: 16px;
    margin-bottom: 20px;
}
.gradio-btn {
    background-color: #007bff;
    color: #ffffff;
    border: none;
    border-radius: 8px;
    padding: 12px 24px;
    font-size: 16px;
    cursor: pointer;
}
.gradio-btn:hover {
    background-color: #0056b3;
}
.gradio-btn:focus {
    outline: none;
    box-shadow: 0 0 0 2px rgba(0, 123, 255, 0.5);
}
"""

# Function to get responses from Watson API
def get_response(user_input, history):
    # Append the new user input to the history
    history.append(("User", user_input))

    # Construct prompt with the entire history
    history_text = '\n'.join(f'{speaker}: {message}' for speaker, message in history)
    prompt_input = f"{context}\n{history_text}\nBot:"

    # Get response from the model
    generated_response = model.generate_text(prompt=prompt_input, guardrails=True)

    # Append the model's response to the history
    history.append(("Bot", generated_response))

    # Return the updated history
    return history, history

# Create a Gradio interface with a chat-like experience
interface = gr.Interface(
    fn=get_response,
    inputs=[
        gr.Textbox(label="Enter your message", placeholder="Type your message here..."),
        gr.State([])  # Initialize chat history as an empty list
    ],
    outputs=[
        gr.Chatbot(label="Chatbot"),  # Use Chatbot component for a WhatsApp-like interface
        gr.State([])  # Use State to maintain chat history
    ],
    title="Dental Clinic Chatbot",
    description="Chat with our bot about Bright Smiles Dental Clinic.",
    css=custom_css,  # Apply custom CSS
    allow_flagging=False  # Disable flagging option
)

# Launch the interface
if __name__ == "__main__":
    interface.launch()
