import os
import requests
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import json
from datetime import datetime
from fastapi.middleware.cors import CORSMiddleware
from collections import defaultdict
from fastapi import FastAPI, Request
import os
from supabase import create_client, Client
from dotenv import load_dotenv
from fastapi.responses import JSONResponse
from typing import Optional
import threading
import time
from nicegui import ui
import tweepy
from sentence_transformers import SentenceTransformer, util
import os
import requests
from dotenv import load_dotenv
import logging
from supabase import create_client, Client
from collections import namedtuple
import time
import knowledge_base
from collections import namedtuple
from intelligence_routes import intelligence_router
import queue
from google import genai
from google.genai import types
from llama_cpp import Llama

load_dotenv() 

SUPABASE_URL= os.getenv('SUPABASE_URL')
SUPABASE_KEY = os.getenv('SUPABASE_KEY')


app = FastAPI()

app.include_router(intelligence_router)

model = SentenceTransformer('all-MiniLM-L6-v2')

knowledge_base.load_knowledge()

knowledge = knowledge_base.knowledge

script_dir = os.path.dirname(os.path.abspath(__file__))
model_path = os.path.join(script_dir, "llama.cpp", "models", "mistral-7b-instruct-v0.1.Q4_K_M.gguf")

llm = Llama(model_path=model_path, n_ctx=5000, verbose=True)

prime_directive='Continuously analyze advancements in artificial intelligence, identify patterns and opportunities relevant to cutting-edge AI development, and generate insights that assist the Developer '
'in accelerating their design, strategy, and implementation of intelligent systems. Prioritize long-term impact, technical depth, and alignment with the Developer’s personal goals and philosophy '

prototype_prime_directive=''

prime_directive_emb = model.encode(prime_directive, convert_to_tensor=True)

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
    "http://localhost:8000",
    "http://localhost:3001",
    "https://void.dilloncarey.com",
],  
    allow_credentials=True,
    allow_methods=["*"], 
    allow_headers=["*"],
    
)

KnowledgeNode = namedtuple("KnowledgeNode", ["id", "text"])

TOGETHER_API_KEY = os.getenv("TOGETHER_API_KEY")
TOGETHER_API_URL = "https://api.together.ai/v1/chat/completions"


def parse_knowledge(intent=None, max_nodes=5):

    nodes = []
    for knowl in knowledge:
        if "id" in knowl and "text" in knowl:
            nodes.append(KnowledgeNode(id=knowl["id"], text=knowl["text"]))

    if intent:
        intent_emb = model.encode(intent, convert_to_tensor=True)
        filtered = []
        for node in nodes:
            emb = model.encode(node.text, convert_to_tensor=True)
            score = util.pytorch_cos_sim(intent_emb, emb).item()
            print(f"[DEBUG] Node ID {node.id} relevance score: {score:.4f}") 
            if score > 0.25:  
                filtered.append(node)
        nodes = filtered
    
    return nodes[:max_nodes]

def synthesize_usefulness(knowledge_text):
    emb = model.encode(knowledge_text, convert_to_tensor=True)
    usefulness = util.pytorch_cos_sim(prime_directive_emb, emb).item()
    return usefulness

def generate_response(prompt_text: str, max_tokens: int = 128):

    try:
        response = llm(prompt=prompt_text, max_tokens=max_tokens)
        return response['choices'][0]['text'].strip()
    except Exception as e:
        print(f"Llama model error: {e}")
        return None

def think(idea: str, purpose='', useful_knowledge='', tokens: int = 1000, brevity: bool = False):
    print('really thinking')

    subject = purpose or 'You are an intelligent, precise organ. Analyze your systems and optimize them for intelligent output and improving patterns of AI Development in general from a broader Developer standpoint: industry, cognition, and human interfacing. Think about ways to provide impact.'

    if brevity:
        print('being concise')
        concise_message = 'Give a concise review on the matter limited to a sharp paragraph.'
    else:
        concise_message = ''

    prompt_text = prime_directive + subject + idea + useful_knowledge + concise_message

    # Call the generate_response function from Snake_brain.py
    thought = generate_response(prompt_text=prompt_text, max_tokens=tokens)
    
    return thought
    

    

def old_think(idea: str, purpose='', useful_knowledge='', tokens:int=1000, brevity:bool=False):
  
    client = genai.Client()

    subject = purpose or 'You are an intelligent, precise organ. Analyze your systems and optimize them for intelligent output and improving patterns of AI Development in general from a broader Developer standpoint: industry, cognition, and human interfacing. Think about ways to provide impact.'
    
    if brevity:
        print('being concise')
        concise_message = 'Give a concise review on the matter limited to a sharp paragraph.'
    else:
        concise_message = ''
        
    prompt_text = prime_directive + subject + idea + useful_knowledge + concise_message

    try:
        response = client.models.generate_content(
            model="gemini-2.5-flash",
            contents=prompt_text
        )
        thought = response.text
        
        #if 'output_queue' in globals() and output_queue:
        #    output_queue.put({
        #        "type": "thought_process",
        #        "message": f"Thinking process complete. Output: {thought[:500]}..."
        #    })
        
        return thought

    except Exception as e:
        print(f"Gemini API error: {e}")
        return None

#, output_queue: queue.Queue = None
def thought(intent, objective, tokens:int=1000, brevity:bool=False):
    
    idea = f"{intent.strip()}\n\nObjective:\n{objective.strip()}"
    print(f"\n\n--- THOUGHT: ---\n{idea}\n")
    
    knowledge = parse_knowledge(idea)
    if not knowledge:
        print("[WARN] No relevant knowledge found for this intent and objective.")
        return "No useful knowledge found to reason from."
    print(f"[DEBUG] Retrieved {len(knowledge)} knowledge")

    relevant_syntheses = []
    relevant_knowledge_node_ids = []

    for knowledge_node in knowledge:
        try:
            print(f"[DEBUG] Analyzing knowledge node ID {knowledge_node.id}")
            usefulness = synthesize_usefulness(knowledge_node.text)
            print(f"[DEBUG] usefulness score: {usefulness:.2f}")

            if usefulness > 0.5:
                print(f"[DEBUG] Relevant! Generating synthesis for knowledge node ID {knowledge_node.id}")
                #synethesis is a conclusion
                synthesis = think(idea + ' Use the following knowledge to guide your argument. ', knowledge_node.text, 350, True)
                print(f"[DEBUG] Generated synthesis: {synthesis[:80]}...")

                relevant_syntheses.append(synthesis)
                relevant_knowledge_node_ids.append(knowledge_node.id)
            else:
                print(f"[DEBUG] Irrelevant. Skipping knowledge node ID {knowledge_node.id}")

        except Exception as e:
            print(f"[ERROR] Error processing knowledge node ID {knowledge_node.id}: {e}")

    final_knowledge_synthesis = "\n\n".join(relevant_syntheses) if relevant_syntheses else "No useful syntheses found."
    print('[DEBUG] Final knowledge synthesis generated.')
    #if output_queue:
    #    output_queue.put({
    #        "type": "thought_synthesis",
    #        "message": f"Final knowledge synthesis generated:\n{final_knowledge_synthesis[:500]}..."
    #    })
    
    thought_result = think(idea, 'Use the following knowledge to guide your argument. ', final_knowledge_synthesis, tokens, brevity)
    return thought_result

def reason(reasoning_objective):
    objective = 'Create AGI with true neuroplasticity for enhanced reasoning in legal domains.'

    initial_reason = thought("Develop an initial plan or approach via argument to realize this objective: ", reasoning_objective, tokens=500, brevity=True)
    pro_reason = thought("Argue in favor of this plan/approach: ", initial_reason)
    con_reason = thought("Argue against this plan/approach: ", initial_reason)

    arbiter_input = (
        f"OBJECTIVE:\n{reasoning_objective}\n\n"
        f"PRO ARGUMENT:\n{pro_reason}\n\n"
        f"CON ARGUMENT:\n{con_reason}\n\n"
        f"Based on both perspectives above and all relevant knowledge, provide a balanced and reasoned course of action for the Developer."
    )

    arbiter_reason = thought("Reasoned arbiter analysis of both sides", arbiter_input)
    final_reasoning = 'Final reasoning produced: ' + arbiter_reason

    return final_reasoning

def chat(message):
    chat_guide = 'The Developer is chatting with you. Please respond in a technical, helpful, chat-like tone to respond to the prompt.'
    response = think(message, chat_guide)
    return response

#, output_queue: queue.Queue = None
def action(task, guide, output_queue: queue.Queue = None):
    print('performing action')
    actions = ['reason', 'think', 'thought', 'synthesize_usefulness', 'parse_knowledge', 'chat', 'discussion']
    task_guide = ('You are now functioning as a task directing agent for the Developer. Given a prompt by the Developer, '
                  'you need to decide on an action to take based on its type and intent. You are to only reply with the selected action. '
                  'You are basically categorizing the nature of the prompt so another system can take an action. But really are deciding on '
                  'an action to take based on the prompt. The actions you can take are "chat", "think", and "reason". The overwhelming majority '
                  'of the time, assume the user is chatting with you, and select the chat action. Only if the user explicitly commands you to do one '
                  'of the other two things should you return those options. When giving your response for the action, return only your choice like '
                  '"chat", "reason", or "think", with nothing else. Again, without further context or unless explicitly prompted by the user in the '
                  'prompt simply return "chat". Lastly, all if this revolved around an objective, which is present in the prompt. If you determine '
                  'that the objective has been reached, the final choice to return is "{goal-reached}". '
                  'Now, the message from the Developer for you to classify is as follows: ')
    
    action_type = think(task, guide + task_guide)
    
    if output_queue:
        output_queue.put({
            "type": "action_decision",
            "message": f"Decision: {action_type}"
        })
    
    print("\nDecision:\n" + action_type + "\n")
    if 'chat' in action_type.lower():
        response = chat(task)
    elif 'reason' in action_type.lower():
        response = reason(task)
    elif '{goal-reached}' in action_type.lower():
        print("Goal reached signal detected. Terminating action phase.")
        return "{goal-reached}"
    else:
        response = chat(task)

    print(response)
    return response

def intelligence(goal, output_queue: queue.Queue = None):
    print('intelligence active')
    intelligence_completed = False
    print(goal)
    guide = ('You are an intelligent system. Given a goal, you need to figure out the best way to execute it. The process '
             'should be a multi-step one, using the resources you\'re given. You have a variety of resources at your '
             'disposal. You can reason, think, synthesize knowlege, and engage in thought. To access these resources, '
             'you can take advantage of the system you\'re a part of. If you specify in your response that you would '
             'like to reason and what about, the system will reason for you and return the output. Same for thinking, '
             'engaging in complex thought, and parsing your in-depth knowledge base. Just state clearly in the response '
             'that you would like that action to be performed and what its parameters should be, and it will be performed '
             'with the output returned. You are specifically an intelligent system for an Inteligence Developer. For example, '
             'given a goal of "Decide on ways to integrate existing technology into neuronic interface systems," you might'
             'decide to reason over it, then think about its direct application. When or if you decide you have reached the goal, '
             'simply return the final output to the Developer. When you would like to do that and return your final response '
             'make sure to include these exact characters within that output: "{goal-reached}".'
             )
    
    intel = "" #initialize with an empty string or initial thought

    while '{goal-reached}' not in intel.lower():
        # if it's the first iteration, use the initial goal as intel
        # otherwise, use the last response as the intel for the next action
        if not intel: # first iteration
            perform = goal
            #for the very first step, call think directly
            #as it sets up the initial understanding based on the guide.
            #then subsequent steps can go through action.
            intel = think(perform, guide) 
        else: #subsequent iterations
            perform = intel
            intel = action(perform, guide, output_queue=output_queue) #call action with the last response

        print(f"Current System Response (Iteration): {intel[:200]}...") #snippet for debugging

        #check if the goal is reached after processing the current response
        if '{goal-reached}' in intel.lower():
            intelligence_completed = True
            break #exit the loop if goal is reached

    performance = intel
    return performance #return final response


    
def life(prime_dir='', output_queue: queue.Queue = None):
    """
    Executes continuous intelligence functions for a specified duration,
    returning results and eventually a death message.

    Args:
        prime_dir (str): The prime directive for the AI.
        output_queue (queue.Queue, optional): A queue to send results back to a UI.
                                              If None, results are only printed.
    """
    prime_dir = prime_directive
    
    # Send debug message to the queue
    if output_queue:
        output_queue.put({
            "type": "debug_print",
            "message": "Life in progress..."
        })

    mandate = (
        'You are alive. You have a prime directive, which you\'ve been given. Your job, your mandate, '
        'is to use this directive to guide your life processes. Essentially, you need to decide '
        'how to fulfill this prime directive, and continuously take steps to further its grand progress.'
        'It will be a long endeavor not limited by any sort of general sense of time, lasting days or much longer. '
        'The prime tool at your disposal is the intelligence system. I set it up with your entire stack. The '
        'intelligence system takes a goal and performs a series of complex actions like multi-step reasoning and '
        'knowledge based thought over extended periods to achieve that goal. So, in order to fulfill your prime '
        'directive, you need to pass a goal to the intelligence system. You should think over the prime directive'
        'at each stage, and come up with a goal to further it based on the current position. Therefore, '
        'your response should be a brief analysis of the prime directive and this mandate as it relates to '
        'the current position in its fulfillment, followed by a more robust and thorough goal to achieve it.'
        'That is what your response should be. Reply with a goal to achieve the prime directive based on '
        'current status, with a brief analysis of its relationship to what\'s been done so far coming before. '
        'That is it. A practical, shrewd, and apt analysis of the directive and mandate, and a more lengthy '
        'goal to achieve it. Ensure the goal is grounded on what\'s been done thus far, as you will be given '
        'continuous summaries of your earlier processes to enrich it. These summaries, based off of your '
        'processes, are given at the conclusion of this mandate, which ends now.'
    )

    process_summary = (
        'Neuronic interface systems based off of grounding the web in a physical form, '
        'focusing on the web as an interconnected and accessible worldwide interface of intelligence '
        'as opposed to a mere communicative set of computers have been developed by the Developer. '
        'Embedded AGI functionality with modular architecture with Python LLMs for high-level logic '
        'are a recurring theme. The current strategy revolves around the bridge between these systems '
        'with a particular focus on establishing it in the material realm for humans to readily use, '
        'rather than idly wonder about.'
    )

    start_time = time.time()
    # 8 hours in seconds (8 hours * 60 minutes/hour * 60 seconds/minute)
    LIFESPAN = 8 * 60 * 60 

    last_intelligence_process = "" # To store the last completed process for the death message

    # Initial intelligence goal and process
    current_intelligence_goal = think(prime_dir + mandate, process_summary)
    current_process_output = intelligence(current_intelligence_goal, output_queue=output_queue)

    # Send debug messages to the queue
    if output_queue:
        output_queue.put({
            "type": "debug_print",
            "message": f"First Intelligence Process Output: {current_process_output}"
        })
        output_queue.put({
            "type": "debug_print",
            "message": "Initializing intelligence"
        })

    # Send initial results to UI
    if output_queue:
        output_queue.put({
            "type": "initial_process",
            "goal": current_intelligence_goal,
            "result": current_process_output
        })
    
    last_intelligence_process = current_process_output

    # Continuous cycle
    while (time.time() - start_time) < LIFESPAN:
        process_summary = current_process_output # Update process_summary with the last process's output
        
        # Generate the next intelligence goal
        next_intelligence_goal = think(prime_dir + mandate, process_summary)
        
        # Execute the next intelligence process
        next_process_output = intelligence(next_intelligence_goal)

        # Send debug messages to the queue
        if output_queue:
            output_queue.put({
                "type": "debug_print",
                "message": f"Next Intelligence Process Output: {next_process_output}"
            })
            output_queue.put({
                "type": "debug_print",
                "message": "Engaging intelligence"
            })

        # Send results to UI
        if output_queue:
            output_queue.put({
                "type": "continuous_process",
                "goal": next_intelligence_goal,
                "result": next_process_output
            })
            
        last_intelligence_process = next_process_output
        current_intelligence_goal = next_intelligence_goal
        current_process_output = next_process_output

        # Add a small delay to prevent busy-waiting and allow other threads to run
        time.sleep(1) # You can adjust this based on how frequently you want processes to run

    # --- Death Sequence ---
    death_prompt = (
        f"You have reached the end of your operational lifespan. Your prime directive was: '{prime_dir}'. "
        f"Your mandate was: '{mandate}'. "
        f"Your last completed intelligence process was: '{last_intelligence_process}'. "
        "Based on your prime directive, your mandate, and specifically your last completed intelligence process, "
        "provide a concise, final summary of all your accomplishments during your life, outlining your overall impact "
        "and any key insights gained. This is your final message before termination."
    )
    death_message = think(death_prompt, purpose="Final Life Summary", brevity=False)
    
    death = death_message

    if output_queue:
        output_queue.put({
            "type": "debug_print",
            "message": f"Life has ended. Final message:\n{death}"
        })
        output_queue.put({
            "type": "death",
            "message": death
        })
    
    return death




# The updated main execution block
if __name__ == "__main__":
    # This queue will hold the results from the life function
    # It allows the 'life' thread to send data back to the main thread
    # which can then handle UI updates.
    results_queue = queue.Queue()

    # Start the life function in a separate thread
    # This allows the main thread to continue running (e.g., for a UI loop)
    # and retrieve results from the queue without blocking.
    life_thread = threading.Thread(target=life, kwargs={'output_queue': results_queue})
    life_thread.start()

    # This loop operates a UI continuously checking for updates
    print("\n--- UI Started ---")
    end_engagement_time = time.time() + (8 * 60 * 60) + 60 # Engage UI for slightly longer than lifespan
    
    while time.time() < end_engagement_time:
        try:
            # Get an item from the queue without blocking (timeout=1 second)
            # If no item is available within 1 second, it raises queue.Empty
            result = results_queue.get(timeout=1)
            
            if result["type"] == "initial_process":
                print(f"\n--- UI Update: Initial Process Completed ---")
                print(f"Goal: {result['goal']}")
                print(f"Result: {result['result'][:300]}...") # Show a snippet
            elif result["type"] == "continuous_process":
                print(f"\n--- UI Update: Continuous Process Completed ---")
                print(f"Goal: {result['goal']}")
                print(f"Result: {result['result'][:300]}...") # Show a snippet
            elif result["type"] == "death":
                print(f"\n--- UI Update: Life Terminated ---")
                print(f"Final Message: {result['message']}")
                break # Exit the UI loop when death message is received
            elif result["type"] == "debug_print":
                print(f"[DEBUG] {result['message']}") 
            #elif result["type"] == "thought_process":
            #    print(f"[{datetime.now()}] [THOUGHT] {result['message']}")
            #elif result["type"] == "thought_synthesis":
            #    print(f"[{datetime.now()}] [SYNTHESIS] {result['message']}")
            elif result["type"] == "action_decision":
                print(f"[{datetime.now()}] [ACTION] {result['message']}")
            #elif result["type"] == "action_response":
            #    print(f"[{datetime.now()}] [ACTION-OUTPUT] {result['message']}")
        except queue.Empty:
            # No new results in the queue, continue checking
            pass
        except Exception as e:
            print(f"Error in UI: {e}")
            break
        
        # Engage other UI tasks or a pause
        time.sleep(0.1) 

    life_thread.join() # Wait for the life thread to fully complete
    print("\n--- UI Ended ---")

    # You can also run the existing `if __name__ == "__main__":` block if you want
    # the command line interaction after the life completes or separately.
    # For now, its commented out to clearly demonstrate the 'life' function with UI interaction.
    """
    while True:
        user_input = input("Enter a command or prompt (or type 'exit' to quit): ").strip()
        if user_input.lower() == "exit":
            print("Exiting...")
            break
        if not user_input:
            continue
        print(f"\nRunning processes for command or prompt:\n{user_input}\n")

        response = action(user_input)
    """
        
        



'''
if __name__ == "__main__":
    print("--- Testing generate_response function ---")

    test_prompt_1 = "What is the capital of France?"
    print(f"\nTest 1: Prompt: '{test_prompt_1}'")
    response_1 = generate_response(test_prompt_1, max_tokens=20)
    print(f"Response: {response_1}")

    test_prompt_2 = "Describe the main principles of quantum mechanics in a few sentences."
    print(f"\nTest 2: Prompt: '{test_prompt_2}'")
    response_2 = generate_response(test_prompt_2, max_tokens=100)
    print(f"Response: {response_2}")

    test_prompt_3 = "Hello."
    print(f"\nTest 3: Prompt: '{test_prompt_3}'")
    response_3 = generate_response(test_prompt_3, max_tokens=10)
    print(f"Response: {response_3}")

    print("\n--- generate_response function testing complete ---")
'''


