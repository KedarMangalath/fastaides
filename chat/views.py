from django.shortcuts import render
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
from django.core.files.base import ContentFile
from django.db import transaction
from django.utils import timezone
from django.conf import settings
from django.core.exceptions import ValidationError
from django.db import DatabaseError, transaction
from .models import ChatSession, Message, Prescription, CustomerDetails, MedicineOrder
from langchain.agents import AgentExecutor, create_openai_functions_agent
from langchain.tools import BaseTool
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from pydantic import BaseModel, Field, PrivateAttr
import json
import base64
from typing import Dict, List, Optional, Any, Union, Tuple
import logging
import os

# Enhanced logging configuration
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
) 
logger = logging.getLogger(__name__)

# Initialize LLMs with error handling
try:
    llm = ChatOpenAI(model_name="gpt-4o-mini", temperature=0.7)
    vision_llm = ChatOpenAI(model_name="gpt-4o-mini", max_tokens=1000)
except Exception as e:
    logger.error(f"Failed to initialize LLMs: {e}")
    raise
 
def log_message(
    chat_session: ChatSession,
    content: str,    
    role: str,
    msg_type: str = 'text',
    metadata: dict = None
) -> Optional[Message]:
    """
    Centralized message logging with metadata
    """
    try:
        return Message.objects.create(
            session=chat_session,
            content=content,
            role=role,
            type=msg_type,
            metadata=metadata or {}
        )
    except Exception as e:
        logger.error(f"Failed to log message: {e}")
        return None

def update_session_context(chat_session: ChatSession, new_context: dict) -> None:
    """
    Update session context while preserving history with UUID handling
    """
    try:
        def convert_uuids(obj):
            if isinstance(obj, dict):
                return {k: str(v) if hasattr(v, 'hex') else convert_uuids(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert_uuids(item) for item in obj]
            elif hasattr(obj, 'hex'):
                return str(obj)
            return obj
            
        new_context = convert_uuids(new_context)
        
        current_context = chat_session.last_context or {}
        if 'history' not in current_context:
            current_context['history'] = []
        
        if current_context.get('current'):
            current_context['history'].append({
                'timestamp': timezone.now().isoformat(),
                'context': current_context['current']
            })
        
        current_context['current'] = new_context
        chat_session.last_context = current_context
        chat_session.save()
        
        logger.debug(f"Successfully updated session context for {chat_session.session_id}")
        
    except Exception as e:
        logger.error(f"Failed to update session context: {str(e)}")
        raise
def validate_image_data(image_data: str) -> tuple[bool, str]:
    """
    Validate image data format and size
    """
    try:
        if not image_data:
            return False, "No image data provided"
            
        # Check base64 format
        try:
            decoded_data = base64.b64decode(image_data)
        except Exception:
            return False, "Invalid image format"
            
        # Check file size (5MB limit)
        if len(decoded_data) > 5 * 1024 * 1024:
            return False, "Image file too large (max 5MB)"
            
        return True, ""
        
    except Exception as e:
        logger.error(f"Image validation error: {str(e)}")
        return False, "Error validating image"

def process_prescription_results(validation_result: dict) -> tuple[str, str, list]:
    """
    Process prescription validation results and generate appropriate response
    """
    try:
        is_valid = validation_result.get('valid', False)
        prescription_type = validation_result.get('type', 'INVALID')
        medicines = validation_result.get('medicines', [])
        
        if is_valid:
            medicine_list = "\n".join(
                f"• {med['name']} - {med.get('dosage', 'as directed')}"
                for med in medicines
            )
            response = (
                f"I've validated your digital prescription and identified the following medications:\n\n"
                f"{medicine_list}\n\n"
                f"Would you like me to help you order these medicines?"
            )
            new_state = 'ORDER_QUANTITY'
        else:
            reason = validation_result.get('reason', 'Unknown error')
            response = (
                f"I apologize, but I cannot process this prescription automatically because: {reason}\n\n"
                "Our team will review your prescription and contact you shortly. "
                "Alternatively, you can upload a typed English prescription for immediate processing."
            )
            new_state = 'MANUAL_REVIEW'
            
        return response, new_state, medicines
        
    except Exception as e:
        logger.error(f"Error processing validation results: {str(e)}")
        raise

def verify_quantities(
    medicines: List[Dict[str, Any]],
    quantities: Dict[str, int]
) -> Tuple[bool, str]:
    """
    Verify medicine quantities against prescribed duration
    """
    try:
        for medicine in medicines:
            name = medicine['name']
            if name not in quantities:
                return False, f"Missing quantity for {name}"
            
            ordered_quantity = quantities[name]
            if not isinstance(ordered_quantity, (int, float)) or ordered_quantity <= 0:
                return False, f"Invalid quantity for {name}"
            
            # Default to 14 days if suggested_quantity not found
            suggested_quantity = 14
            if ordered_quantity > suggested_quantity:
                return False, f"Quantity for {name} exceeds maximum allowed ({suggested_quantity} units)"

        return True, ""
    except Exception as e:
        logger.error(f"Error verifying quantities: {e}")
        return False, "Error verifying medicine quantities"
    
def get_medicine_suggestions(prescription: Prescription) -> str:
    """
    Generate suggested quantities based on prescription
    """
    try:
        if not prescription or not prescription.extracted_data:
            return "No valid prescription data found"

        medicines = prescription.extracted_data
        suggestions = ["Recommended quantities based on your prescription:"]
        
        for medicine in medicines:
            name = medicine['name']
            dosage = medicine.get('dosage', 'as directed')
            duration = medicine.get('days_supply', 14)
            suggested_quantity = medicine.get('suggested_quantity', 14)
            
            suggestions.append(
                f"\n- {name}"
                f"\n  Suggested quantity: {suggested_quantity} units"
                f"\n  Based on: {dosage} for {duration} days"
            )
        
        return "\n".join(suggestions)
    except Exception as e:
        logger.error(f"Error generating medicine suggestions: {e}")
        return "Error generating medicine suggestions. Please try again."

class MedicalInfoTool(BaseTool):
    name: str = "medical_information"
    description: str = """Use this tool for answering general medical questions. 
    Always include medical disclaimers in responses."""
    
    def _run(self, query: str) -> str:
        try:
            logger.debug(f"Processing medical query: {query}")
            response = llm.invoke([
                SystemMessage(content="""You are Fastaides medical information provider. 
                Give accurate, short basic and well-sourced information, with appropriate disclaimers."""),
                HumanMessage(content=query)
            ])
            
            if not response or not response.content:
                logger.error("Empty response received from LLM")
                return ("I apologize, but I couldn't process your medical question. "
                       "Please try rephrasing your question.")
            
            return (f"{response.content}\n\nDisclaimer: This information is for educational "
                   "purposes only and should not replace professional medical advice.")
        except Exception as e:
            logger.exception("Error in medical info tool")
            return ("I encountered an error while processing your medical question. "
                   "Please try again or rephrase your question.")

    async def _arun(self, query: str) -> str:
        raise NotImplementedError("Async not implemented")

class PrescriptionValidationTool(BaseTool):
    name: str = "validate_prescription"
    description: str = """Use this tool to validate if an image shows a valid typed English prescription and extract medicine information."""
    _logger: logging.Logger = PrivateAttr()

    def __init__(self):
        super().__init__()
        self._logger = logging.getLogger(__name__)

    def _run(self, image_data: str) -> Dict[str, Union[bool, str, List[Dict[str, Any]]]]:
        try:
            self._logger.debug("Starting prescription validation")
            
            # Validate image data
            try:
                decoded_image = base64.b64decode(image_data)
                if len(decoded_image) > 5 * 1024 * 1024:  # 5MB limit
                    return {
                        "valid": False,
                        "type": "INVALID",
                        "reason": "Image file too large (max 5MB)",
                        "medicines": []
                    }
            except:
                return {
                    "valid": False,
                    "type": "INVALID", 
                    "reason": "Invalid image format",
                    "medicines": []
                }

            messages = [
                HumanMessage(content=[
                    {
                        "type": "text",
                        "text": """Analyze this prescription image and return a JSON response with these exact fields:
{
    "is_typed": true if clearly computer-generated/typed, false otherwise,
    "has_structured_layout": true if organized in clear sections,
    "hospital_details": true if contains letterhead/hospital info,
    "medicines": [
        {
            "name": "medicine name",
            "dosage": "dosage instructions",
            "duration": "duration in days",
            "quantity": "prescribed quantity"
        }
    ]
}"""
                    },
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{image_data}"
                        }
                    }
                ])
            ]

            response = vision_llm.invoke(messages)
            
            try:
                # Extract JSON from response
                json_start = response.content.find('{')
                json_end = response.content.rfind('}') + 1
                if json_start < 0 or json_end <= json_start:
                    raise ValueError("No valid JSON found in response")
                
                analysis = json.loads(response.content[json_start:json_end])
                
                # Validate prescription format
                is_valid = (
                    analysis.get("is_typed", False) and 
                    analysis.get("has_structured_layout", False) and
                    analysis.get("hospital_details", False)
                )
                
                medicines = analysis.get("medicines", [])
                if is_valid and not medicines:
                    is_valid = False
                    analysis["reason"] = "No medicines found in prescription"
                
                # Process medicines
                processed_medicines = []
                for medicine in medicines:
                    name = medicine.get('name', '').strip()
                    if not name:
                        continue
                        
                    # Process dosage
                    dosage = medicine.get('dosage', '').lower()
                    daily_doses = 1
                    if any(term in dosage for term in ['twice', '2 times', 'two times', 'bid', '2x']):
                        daily_doses = 2
                    elif any(term in dosage for term in ['three times', '3 times', 'tid', '3x']):
                        daily_doses = 3
                    elif any(term in dosage for term in ['four times', '4 times', 'qid', '4x']):
                        daily_doses = 4
                    
                    # Process duration with fallback
                    try:
                        duration_str = medicine.get('duration', '14')
                        duration_num = ''.join(filter(str.isdigit, duration_str)) or '14'
                        days_supply = min(int(duration_num), 14)  # Cap at 14 days
                    except (ValueError, TypeError):
                        days_supply = 14
                    
                    # Calculate suggested quantity
                    suggested_quantity = min(daily_doses * days_supply, daily_doses * 14)
                    
                    processed_medicines.append({
                        'name': name,
                        'dosage': dosage,
                        'days_supply': days_supply,
                        'daily_doses': daily_doses,
                        'suggested_quantity': suggested_quantity,
                        'original_text': medicine.get('dosage', '')
                    })
                
                result = {
                    "valid": is_valid,
                    "type": "DIGITAL" if is_valid else "HANDWRITTEN",
                    "reason": (
                        "Valid computer-generated prescription" if is_valid
                        else "Handwritten or invalid prescription format"
                    ),
                    "medicines": processed_medicines,
                    "analysis": analysis
                }
                
                self._logger.debug(f"Validation result: {result}")
                return result
                
            except json.JSONDecodeError as e:
                self._logger.error(f"JSON parsing error: {e}")
                return {
                    "valid": False,
                    "type": "INVALID",
                    "reason": "Unable to parse prescription format",
                    "medicines": []
                }

        except Exception as e:
            self._logger.exception("Error in prescription validation")
            return {
                "valid": False,
                "type": "INVALID",
                "reason": str(e),
                "medicines": []
            }

    async def _arun(self, image_data: str) -> Dict[str, Any]:
        raise NotImplementedError("Async not implemented")    
        
class MedicineOrderTool(BaseTool):
    name: str = "process_medicine_order"
    description: str = """Tool for processing medicine orders from valid prescriptions. 
    Required format: {
        "session_id": "session identifier",
        "medicines": [list of medicines],
        "quantities": {"medicine_name": quantity},
        "customer_info": {
            "name": "customer name",
            "phone": "phone number",
            "address": "delivery address",
            "email": "optional email"
        }
    }"""
    
    def _run(self, order_input: Union[str, Dict]) -> str:
        try:
            # Handle input parsing
            if isinstance(order_input, str):
                try:
                    order_details = json.loads(order_input)
                except json.JSONDecodeError:
                    return "Error: Invalid order format. Please provide proper order details."
            else:
                order_details = order_input

            # Validate required fields
            required_fields = ['session_id', 'medicines', 'quantities', 'customer_info']
            missing_fields = [field for field in required_fields if field not in order_details]
            if missing_fields:
                return f"Error: Missing required information: {', '.join(missing_fields)}"

            # Process order with transaction
            with transaction.atomic():
                try:
                    session = ChatSession.objects.get(session_id=order_details['session_id'])
                except ChatSession.DoesNotExist:
                    return "Error: Invalid session. Please try again."

                # Verify prescription
                try:
                    prescription = session.prescriptions.filter(
                        status='COMPLETED'
                    ).latest('created_at')
                except Prescription.DoesNotExist:
                    return "Error: No valid prescription found. Please upload a prescription first."

                # Validate quantities
                medicines = order_details['medicines']
                quantities = order_details['quantities']
                customer_info = order_details['customer_info']

                # Verify customer information
                required_customer_info = ['name', 'phone', 'address']
                missing_customer_info = [field for field in required_customer_info 
                                      if not customer_info.get(field)]
                if missing_customer_info:
                    return f"Error: Missing customer information: {', '.join(missing_customer_info)}"

                # Verify quantities
                is_valid, error_msg = verify_quantities(medicines, quantities)
                if not is_valid:
                    return f"Error: {error_msg}"

                # Create customer record
                customer = CustomerDetails.objects.create(
                    session=session,
                    name=customer_info['name'],
                    phone=customer_info['phone'],
                    email=customer_info.get('email', ''),
                    address=customer_info['address']
                )

                # Create order record
                order = MedicineOrder.objects.create(
                    session=session,
                    customer=customer,
                    prescription=prescription,
                    status='CONFIRMED',
                    order_details={
                        'medicines': medicines,
                        'quantities': quantities,
                        'total_items': sum(quantities.values()),
                        'prescription_ref': prescription.reference_number
                    },
                    metadata={
                        'order_time': timezone.now().isoformat(),
                        'delivery_instructions': customer_info.get('delivery_instructions'),
                        'order_source': 'chatbot'
                    }
                )

                # Format response
                medicine_list = "\n".join(
                    f"- {med['name']}: {quantities.get(med['name'], 'as prescribed')} units"
                    for med in medicines
                )

                return (
                    f"Order #{order.id} has been confirmed!\n\n"
                    f"Medicines ordered:\n{medicine_list}\n\n"
                    f"Delivery Details:\n"
                    f"Name: {customer.name}\n"
                    f"Phone: {customer.phone}\n"
                    f"Address: {customer.address}\n\n"
                    "Our team will verify availability and contact you shortly with delivery details."
                )

        except Exception as e:
            logger.exception("Error processing order")
            return f"Error processing order: {str(e)}. Please try again or contact support."

    async def _arun(self, order_details: Dict[str, Any]) -> str:
        raise NotImplementedError("Async not implemented")
    
def chat_view(request):
    """Render chat interface"""
    return render(request, 'chat/chat.html')

@csrf_exempt 
def process_message(request):
    """Process incoming chat messages"""
    if request.method != 'POST':
        return JsonResponse({'error': 'Invalid request method'}, status=405)
    
    try:
        data = json.loads(request.body)
        message_type = data.get('type', 'text')
        session_id = request.session.get('chat_session_id')
        
        logger.debug(f"Processing message type: {message_type}")
        
        # Session handling with better initialization
        if not session_id:
            chat_session = ChatSession.objects.create()
            chat_session.state = 'CHAT'  # Explicit state setting
            chat_session.save()
            request.session['chat_session_id'] = str(chat_session.session_id)
            logger.debug(f"Created new chat session: {chat_session.session_id}")
        else:
            try:
                chat_session = ChatSession.objects.get(session_id=session_id)
                if not chat_session.state:
                    chat_session.state = 'CHAT'
                    chat_session.save()
            except ChatSession.DoesNotExist:
                # Handle invalid session by creating new one
                chat_session = ChatSession.objects.create(state='CHAT')
                request.session['chat_session_id'] = str(chat_session.session_id)
                logger.debug(f"Created replacement session: {chat_session.session_id}")
        
        # Process message based on type
        if message_type == 'prescription':
            return handle_prescription_upload(data, chat_session)
        elif message_type == 'text':
            return handle_text_message(data, chat_session)
        
        return JsonResponse({'error': 'Invalid message type'}, status=400)
        
    except json.JSONDecodeError:
        logger.error("Invalid JSON in request body")
        return JsonResponse({'error': 'Invalid JSON format'}, status=400)
    except Exception as e:
        logger.exception("Error processing message")
        return JsonResponse({'error': str(e)}, status=500)
    

def handle_text_message(data: Dict[str, Any], chat_session: ChatSession) -> JsonResponse:
    try:
        message = data.get('message', '').strip()
        if not message:
            return JsonResponse({'error': 'Empty message'}, status=400)
        
        logger.debug(f"Processing message: {message[:100]}...")
        
        # Detect if this is a greeting/initial message
        greeting_words = {'hi', 'hello', 'hey', 'good morning', 'good afternoon', 'good evening'}
        is_greeting = any(word in message.lower() for word in greeting_words)
        
        # Log user message
        log_message(
            chat_session=chat_session,
            content=message,
            role='user',
            metadata={'state': chat_session.state, 'is_greeting': is_greeting}
        )

        # For greetings, ensure we're in CHAT state
        if is_greeting:
            chat_session.state = 'CHAT'
            chat_session.save()

        # Get chat history
        messages = list(chat_session.messages.order_by('timestamp'))[-10:]
        chat_history = [
            AIMessage(content=msg.content) if msg.role == 'assistant' 
            else HumanMessage(content=msg.content)
            for msg in messages
        ]

        # Get current prescription context if any
        prescription_context = chat_session.get_prescription_context()
        
        # Prepare context for the agent
        context = {
            "session_id": str(chat_session.session_id),
            "current_state": chat_session.state,
            "has_valid_prescription": bool(prescription_context),
            "prescription_details": prescription_context,
            "last_context": chat_session.last_context,
            "is_greeting": is_greeting
        }

        # Try processing with retries
        max_retries = 3
        last_error = None
        
        for attempt in range(max_retries):
            try:
                # Process message with agent
                response = agent_executor.invoke({
                    "input": message,
                    "chat_history": chat_history,
                    "context": context
                })
                
                response_content = response.get('output', '')
                if response_content and isinstance(response_content, str):
                    # Determine new state based on response
                    new_state = determine_state(response_content, chat_session)
                    
                    # Log assistant response
                    log_message(
                        chat_session=chat_session,
                        content=response_content,
                        role='assistant',
                        metadata={
                            'state': new_state,
                            'is_greeting': is_greeting,
                            'prescription_id': prescription_context.get('prescription_id') if prescription_context else None
                        }
                    )
                    
                    # Update session state
                    chat_session.state = new_state
                    chat_session.save()
                    
                    return JsonResponse({
                        'message': response_content,
                        'state': new_state
                    })
                else:
                    last_error = ValueError("Empty or invalid response")
                    continue
                    
            except Exception as e:
                last_error = e
                logger.error(f"Error on attempt {attempt + 1}: {str(e)}")
                if attempt == max_retries - 1:
                    raise last_error

        # If we get here, all retries failed
        raise last_error or ValueError("Failed to get valid response")

    except Exception as e:
        logger.exception("Error handling text message")
        
        # Generate appropriate fallback response
        fallback_response = create_fallback_response(message)
        
        # Log fallback response
        log_message(
            chat_session=chat_session,
            content=fallback_response,
            role='assistant',
            metadata={
                'error': str(e),
                'state': chat_session.state or 'CHAT',
                'is_greeting': is_greeting if 'is_greeting' in locals() else False
            }
        )
        
        # Ensure valid state
        if not chat_session.state:
            chat_session.state = 'CHAT'
            chat_session.save()
            
        return JsonResponse({
            'message': fallback_response,
            'state': chat_session.state
        })
            
@transaction.atomic
def handle_prescription_upload(data: Dict[str, Any], chat_session: ChatSession) -> JsonResponse:
    """
    Handle prescription image upload and processing with improved error handling
    """
    logger.info(f"Starting prescription upload for session {chat_session.session_id}")
    
    try:
        # Extract and validate image data
        if 'image' not in data:
            return JsonResponse({'error': 'No image provided'}, status=400)
            
        image_parts = data['image'].split('base64,')
        if len(image_parts) != 2:
            return JsonResponse({'error': 'Invalid image format'}, status=400)
            
        image_data = image_parts[1]
        is_valid_image, error_message = validate_image_data(image_data)
        
        if not is_valid_image:
            return JsonResponse({'error': error_message}, status=400)
        
        # Process prescription
        validation_tool = PrescriptionValidationTool()
        validation_result = validation_tool._run(image_data)
        
        # Create prescription record
        prescription = Prescription.objects.create(
            session=chat_session,
            status='PROCESSING',
            prescription_type=validation_result['type'],
            extracted_data=validation_result.get('medicines', []),
            validation_errors=None if validation_result['valid'] else {
                'reason': validation_result['reason'],
                'details': validation_result.get('analysis', {})
            }
        )
        
        # Save image file
        try:
            prescription.image.save(
                f'prescription_{prescription.reference_number}.jpg',
                ContentFile(base64.b64decode(image_data))
            )
        except Exception as e:
            logger.error(f"Error saving prescription image: {str(e)}")
            raise
            
        # Process validation results
        response_message, new_state, medicines = process_prescription_results(validation_result)
        
        # Update prescription status
        prescription.status = 'COMPLETED' if validation_result['valid'] else 'MANUAL_REVIEW'
        prescription.save()
        
        # Update session context
        if validation_result['valid']:
            update_session_context(chat_session, {
                'prescription_id': str(prescription.id),
                'medicines': medicines,
                'reference': str(prescription.reference_number)
            })
        
        # Log the interaction
        log_message(
            chat_session=chat_session,
            content=response_message,
            role='assistant',
            msg_type='prescription',
            metadata={
                'prescription_id': str(prescription.id),
                'validation_result': {
                    'valid': validation_result['valid'],
                    'type': validation_result['type'],
                    'medicine_count': len(medicines)
                },
                'status': prescription.status
            }
        )
        
        # Update session state
        chat_session.state = new_state
        chat_session.save()
        
        # Prepare response data
        response_data = {
            'message': response_message,
            'state': new_state,
            'prescription_data': {
                'id': str(prescription.id),
                'type': prescription.prescription_type,
                'reference': str(prescription.reference_number),
                'medicines': prescription.extracted_data
            }
        }
        
        logger.info(f"Successfully processed prescription for session {chat_session.session_id}")
        return JsonResponse(response_data)
        
    except ValidationError as e:
        logger.error(f"Validation error: {str(e)}")
        return JsonResponse({
            'error': 'Invalid prescription data',
            'details': str(e)
        }, status=400)
        
    except DatabaseError as e:
        logger.error(f"Database error: {str(e)}")
        return JsonResponse({
            'error': 'Error saving prescription data',
            'details': 'Please try again later'
        }, status=500)
        
    except Exception as e:
        logger.exception("Unexpected error processing prescription")
        return JsonResponse({
            'error': 'Error processing prescription',
            'details': 'An unexpected error occurred'
        }, status=500)
        
def handle_order_confirmation(chat_session: ChatSession) -> JsonResponse:
    """Handle final order confirmation and processing"""
    try:
        context = chat_session.last_context.get('current', {})
        if not context:
            return JsonResponse({
                'error': 'No active order context found'
            }, status=400)

        medicines = context.get('medicines', [])
        if not medicines:
            return JsonResponse({
                'error': 'No medicines found in order',
                'state': 'ORDER_QUANTITY'
            }, status=400)

        # Create properly formatted medicines list and quantities dict
        formatted_medicines = []
        quantities = {}
        for medicine in medicines:
            if isinstance(medicine, dict):
                name = medicine.get('name')
                quantity = medicine.get('quantity')
                if name and quantity:
                    formatted_medicines.append({
                        'name': name,
                        'dosage': medicine.get('dosage', 'as directed')
                    })
                    quantities[name] = quantity

        if not quantities:
            return JsonResponse({
                'error': 'Missing medicine quantities',
                'state': 'ORDER_QUANTITY'
            }, status=400)

        # Create order input
        order_input = {
            'session_id': str(chat_session.session_id),
            'medicines': formatted_medicines,
            'quantities': quantities,
            'customer_info': context.get('customer_info', {})
        }

        # Validate customer info
        customer_info = order_input['customer_info']
        required_fields = ['name', 'phone', 'address']
        missing_fields = [field for field in required_fields if not customer_info.get(field)]
        if missing_fields:
            return JsonResponse({
                'error': f'Missing customer information: {", ".join(missing_fields)}',
                'state': 'COLLECT_INFO'
            }, status=400)

        order_tool = MedicineOrderTool()
        response = order_tool._run(order_input)

        if response.startswith('Error'):
            return JsonResponse({'error': response}, status=400)

        chat_session.state = 'ORDER_COMPLETE'
        chat_session.save()

        log_message(
            chat_session=chat_session,
            content=response,
            role='assistant',
            msg_type='order',
            metadata={'order_status': 'confirmed'}
        )

        return JsonResponse({
            'message': response,
            'state': 'ORDER_COMPLETE'
        })

    except Exception as e:
        logger.exception("Error in order confirmation")
        return JsonResponse({
            'error': str(e)}, 
            status=500
        )
            
def determine_state(response: str, chat_session: ChatSession) -> str:
    """Determine chat state based on assistant's response"""
    response_lower = response.lower()
    current_state = chat_session.state
    
    # Check for greeting/welcome messages
    greeting_phrases = ['hello', 'hi there', 'how can i assist', 'welcome']
    if any(phrase in response_lower for phrase in greeting_phrases):
        return 'CHAT'
    
    # If currently in ORDER_QUANTITY state, look for clear progress indicators
    if current_state == 'ORDER_QUANTITY':
        # Progress to collecting info if user wants to proceed with order
        if any(phrase in response_lower for phrase in [
            "proceed with order",
            "collect your details",
            "would like to order",
            "let's proceed with the order"
        ]):
            return 'COLLECT_INFO'
        
        # Stay in ORDER_QUANTITY only if explicitly discussing quantity changes
        if any(phrase in response_lower for phrase in [
            "modify quantities",
            "change quantities",
            "specify new quantities"
        ]):
            return 'ORDER_QUANTITY'
            
    # Regular state transitions
    if "upload" in response_lower and "prescription" in response_lower:
        return 'AWAITING_PRESCRIPTION'
        
    if "delivery details" in response_lower or "shipping information" in response_lower:
        return 'COLLECT_INFO'
        
    if "confirm" in response_lower and "order" in response_lower:
        return 'CONFIRM_ORDER'
        
    if "order" in response_lower and ("confirmed" in response_lower or "complete" in response_lower):
        return 'ORDER_COMPLETE'
        
    if "prescription" in response_lower and any(word in response_lower for word in ['invalid', 'cannot process', 'handwritten']):
        return 'MANUAL_REVIEW'
            
    return current_state if current_state else 'CHAT'

def create_fallback_response(message: str) -> str:
    """Create appropriate fallback response when regular processing fails"""
    try:
        message_lower = message.lower()
        
        # Handle greetings
        greeting_words = {'hi', 'hello', 'hey', 'good morning', 'good afternoon', 'good evening'}
        if any(word in message_lower for word in greeting_words):
            return ("Hello! How can I assist you today? If you have any medical questions "
                   "or need help with a prescription, feel free to ask!")
        
        # For medical questions, try using the medical info tool
        if any(word in message_lower for word in ['what', 'how', 'can', 'is', 'are']):
            try:
                medical_tool = MedicalInfoTool()
                tool_response = medical_tool._run(message)
                if tool_response:
                    return tool_response
            except Exception:
                pass
        
        # For prescription/order related queries
        if any(word in message_lower for word in ['prescription', 'medicine', 'order']):
            return (
                "I apologize for the confusion. To process any medication orders, I need a valid digital prescription. "
                "Please share your prescription and I'll be happy to assist you."
            )
            
        # Default response
        return (
            "I apologize, but I'm having trouble understanding your request right now. "
            "Could you please rephrase your question? I'm here to help with medical questions "
            "and prescription processing."
        )
        
    except Exception:
        # Ultimate fallback
        return (
            "I apologize for the technical difficulty. Please try asking your question again, "
            "or contact our support team if the issue persists."
        )
    
# System prompt and agent setup
SYSTEM_PROMPT = """You are Fastaides, a professional medical assistant chatbot.

Your capabilities and workflow:
1. Answer medical questions very shortly with proper disclaimers, never suggest medicine alternatives, and never answer non-medical questions.
2. Process and validate prescriptions
3. Handle medicine orders properly using the required tools. 
5. Use this flow when reciving a prescription:
    - Ask for the medicine quantities
    - Ask for name
    - Ask for Address
    - Send confirmation message and that team will get back to them
When a user sends "Help," always respond with a message detailing what you can assist them with, presented in a clear, bulleted list. For example:

"Hi! Here's how I can assist you:

Answer your medical questions: Get instant answers to general medical queries.
Order medicines: Quickly place orders for your required medicines.
Assist with existing orders: Check the status or update your current orders.
Provide guidance: Get advice on how to use the platform or address any issues you’re facing."
Make sure the tone is helpful and professional.    


{context}

Chat History:
{chat_history}

Current Query:
{input}"""


# Create prompt template and agent
prompt = ChatPromptTemplate.from_messages([
    ("system", SYSTEM_PROMPT),
    MessagesPlaceholder(variable_name="chat_history"),
    ("human", "{input}"),
    MessagesPlaceholder(variable_name="agent_scratchpad"),
])

tools = [MedicalInfoTool(), PrescriptionValidationTool(), MedicineOrderTool()]

try:
    agent = create_openai_functions_agent(llm, tools, prompt)
    agent_executor = AgentExecutor(
        agent=agent,
        tools=tools,
        verbose=True,
        handle_parsing_errors=True,
        max_iterations=3
    )
except Exception as e:
    logger.error(f"Failed to create agent: {e}")
    raise
