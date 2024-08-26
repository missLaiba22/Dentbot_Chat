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
context = """Bright Smiles Dental Clinic: Comprehensive Patient Information
Patient FAQs
What types of dental treatments do you offer at Bright Smiles Dental Clinic?
At Bright Smiles Dental Clinic, we provide a comprehensive range of dental treatments including:
Routine Check-ups: Thorough examination of teeth and gums, with X-rays if needed.
Cleanings: Professional dental cleaning, including scaling and polishing.
Fillings: Treatment for cavities using composite or amalgam materials.
Extractions: Removal of decayed or damaged teeth.
Root Canals: Treatment for infected or damaged tooth roots.
Crowns and Bridges: Restoration for damaged or missing teeth.
Teeth Whitening: Cosmetic procedures to brighten your smile.
Orthodontics: Braces and clear aligners for teeth straightening.
Periodontal Treatment: Addressing gum disease and related conditions.
Emergency Care: Immediate treatment for severe pain, trauma, or infections.
How can I book an appointment?
Appointments at Bright Smiles Dental Clinic can be scheduled through:
Phone: Call us at +1-800-555-1234.
Email: Send a request to contact@brightsmilesdental.com.
Online: Use our user-friendly online booking system available on our website: www.brightsmilesdental.com.
What are your clinic's operating hours?
Our clinic is open during the following hours:
Monday to Friday: 9:00 AM - 6:00 PM
Saturday: 10:00 AM - 3:00 PM
Sunday: Closed
What payment options do you accept?
We offer several payment options:
Credit/Debit cards
Cash
Financing plans
We also accept most major dental insurance plans. Please verify your specific coverage details with us.
What is your cancellation policy?
To avoid cancellation fees, please cancel or reschedule your appointment at least 24 hours in advance.
Case Studies
Case Study 1: Teeth Whitening Transformation
Patient: Sarah M.
Issue: Severe staining from coffee and tea consumption.
Treatment: Professional teeth whitening session.
Outcome: Sarah’s teeth became significantly whiter, enhancing her smile and boosting her confidence. She expressed great satisfaction with the outcome and the comfort of the procedure.
Case Study 2: Successful Orthodontic Alignment
Patient: John R.
Issue: Crooked teeth requiring correction.
Treatment: Clear aligners.
Outcome: John achieved a beautifully straight smile in just 12 months. He appreciated the discreet nature of the aligners and the effectiveness of the treatment.
Case Study 3: Emergency Root Canal Relief
Patient: Emily L.
Issue: Severe toothache due to an abscess.
Treatment: Root canal therapy.
Outcome: The procedure provided immediate relief from pain, and Emily was very pleased with the professional care she received and the resolution of her dental emergency.
Feedback and Surveys
Patient Feedback Highlights
Positive Comments:
Patients frequently commend our friendly and professional staff.
Many appreciate the thorough and effective treatment provided.
Suggestions for Improvement:
Some patients have requested extended clinic hours.
A few have suggested more flexible appointment scheduling options.
Survey Results Summary
1. Overall Quality of Care:
Excellent: 75%
Good: 20%
Average: 5%
2. Satisfaction with Appointment Scheduling:
Very Satisfied: 60%
Satisfied: 25%
Neutral: 10%
Dissatisfied: 5%
3. Likelihood to Recommend Our Clinic:
Definitely: 80%
Probably: 15%
Not Sure: 5%
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

custom_css = """
/* Custom CSS for a modern dark navy blue theme with professional accents */
/* Overall container styling */
.gradio-container {
    font-family: 'Arial', sans-serif;
    background-color: #FFCC99; 
    border-radius: 12px;
    padding: 25px;
    max-width: 850px;
    margin: 0 auto;
    box-shadow: 0 5px 10px rgba(0, 0, 0, 0.3);
    color: #f5f5f5; /* Light text color for readability */
}
/* Title styling */
.gradio-title {
    font-size: 36px;
    font-weight: 700;
    color: #ffffff !important; /* White text for title */
    text-align: center;
    margin-bottom: 20px;
    letter-spacing: 1px;
}
/* Description styling */
.gradio-description {
    font-size: 18px;
    color: #e0e0e0 !important; /* Slightly off-white for description */
    text-align: center;
    margin-bottom: 30px;
    line-height: 1.5;
    font-family: 'Helvetica Neue', sans-serif;
}
/* Chatbot container styling */
.gradio-chatbot {
    background: #1c2d3f; /* Darker navy blue for chatbot area */
    border: 1px solid #34495e;
    border-radius: 12px;
    padding: 15px;
    box-shadow: 0 4px 6px rgba(0, 0, 0, 0.2);
    color: #ffffff;
}
/* Input box styling */
.gradio-input {
    background-color: #ffffff;
    border: 1px solid #ced4da;
    border-radius: 12px;
    padding: 12px;
    font-size: 16px;
    color: #1c2d3f;
    margin-bottom: 20px;
}
/* Button styling */
.gradio-btn {
    background-color: #ffca28;
    color: #1c2d3f;
    border: none;
    border-radius: 10px;
    padding: 12px 20px;
    font-size: 16px;
    cursor: pointer;
    transition: background-color 0.3s ease;
}
.gradio-btn:hover {
    background-color: #ffc107;
}
.gradio-btn:focus {
    outline: none;
    box-shadow: 0 0 0 3px rgba(255, 202, 40, 0.5);
}
/* Chatbot message styling */
.gradio-chatbot .message.user {
    background-color: #ffffff;
    color: #1c2d3f;
    border-radius: 10px;
    padding: 8px 12px;
    margin: 5px 0;
}
.gradio-chatbot .message.bot {
    background-color: #005f73;
    color: #ffffff;
    border-radius: 10px;
    padding: 8px 12px;
    margin: 5px 0;
}
"""
def get_response(patient_name, user_message, state):
    # Unpack state into history and user profile
    history, user_profile = state

    # Emergency information
    emergency_info = {
        "contact": {
            "clinic": "+1-800-123-4567",
            "local_emergency": "911",
            "hospital": "+1-800-987-6543"
        },
        "guidance": {
            "tooth": "If you have a knocked-out tooth, try to place it back in the socket. If that’s not possible, store it in milk and visit the nearest emergency room immediately.",
            "bleeding": "For severe bleeding, apply pressure to the affected area and seek immediate medical attention."
        }
    }

    # Check if the user has already provided their name
    if not user_profile.get('name'):
        if patient_name:
            # Save the patient's name and respond with a greeting
            user_profile['name'] = patient_name
            greeting_message = f"Nice to meet you, {user_profile['name']}! How can I assist you today?"
            history.append(("Bot", greeting_message))
            return history, (history, user_profile)  # Return updated history and combined state
        else:
            # Ask for the user's name if not provided
            prompt_message = "Please provide your name so I can assist you better."
            history.append(("Bot", prompt_message))
            return history, (history, user_profile)  # Return current state without updating

    # Handle emergency queries
    if "emergency" in user_message.lower():
        response = "In case of an emergency, please contact:\n"
        response += f"Clinic Emergency Line: {emergency_info['contact']['clinic']}\n"
        response += f"Local Emergency Services: {emergency_info['contact']['local_emergency']}\n"
        response += f"Nearest Hospital: {emergency_info['contact']['hospital']}\n\n"
        response += "For common dental emergencies:\n"
        response += f"Tooth Knocked Out: {emergency_info['guidance']['tooth']}\n"
        response += f"Severe Bleeding: {emergency_info['guidance']['bleeding']}\n"
        history.append(("Bot", response))
        return history, (history, user_profile)

    # If the user message is empty or too short, return a prompt for more details
    if not user_message or len(user_message.strip()) < 3:
        response = "I didn't quite understand that. Can you please provide more details or ask a specific question?"
        history.append(("Bot", response))
        return history, (history, user_profile)

    # List of keywords or phrases related to the clinic's information
    scope_keywords = ["clinic", "appointment", "treatment", "dentist", "schedule", "hours", "services"]

    # Check if the user message contains any scope keywords
    if not any(keyword in user_message.lower() for keyword in scope_keywords):
        response = "I’m sorry, but I don't have information on that topic. Please ask about our clinic, appointments, treatments, or other related services."
        history.append(("Bot", response))
        return history, (history, user_profile)

    # Append the new user message to the history
    history.append(("User", user_message))

    # Personalize the prompt input with the user's name if available
    personalized_context = f"Hello {user_profile['name']}, " if 'name' in user_profile else ""

    # Construct prompt with the entire history and personalized context
    history_text = '\n'.join(f'{speaker}: {message}' for speaker, message in history)
    prompt_input = f"{personalized_context}\n{history_text}\nBot:"

    # Get response from the model (Ensure the model is loaded and available)
    generated_response = model.generate_text(prompt=prompt_input, guardrails=True)

    # Append the model's response to the history
    history.append(("Bot", generated_response))

    # Return the updated history and combined state (history, user_profile)
    return history, (history, user_profile)


import gradio as gr

# Create a Gradio interface with separate textboxes for name and message
interface = gr.Interface(
    fn=get_response,
    inputs=[
        gr.Textbox(label="Enter your name", placeholder="Type your name here..."),
        gr.Textbox(label="Enter your message", placeholder="Type your message here..."),
        gr.State([[], {}])  # Initialize chat history as an empty list and user profile as an empty dict
    ],
    outputs=[
        gr.Chatbot(label="Chatbot"),  # Use Chatbot component for a WhatsApp-like interface
        gr.State([[], {}])  # Maintain chat history and user profile in a single state
    ],
    title="DENTBOT",
    description="Chat with our bot about Bright Smiles Dental Clinic.",
    css=custom_css,  # Apply custom CSS
    allow_flagging=False  # Disable flagging option
)

# Launch the interface
interface.launch()

