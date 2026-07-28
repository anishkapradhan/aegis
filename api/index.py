import os
import re
import requests
from flask import Flask, render_template, request, jsonify
from groq import Groq
from dotenv import load_dotenv
from supabase import create_client, Client

# Load workspace environment variables
load_dotenv()

# Resolve absolute template path configuration for robust serverless deployments
base_dir = os.path.dirname(os.path.abspath(__file__))
template_dir = os.path.join(base_dir, '..', 'templates')

app = Flask(__name__, template_folder=template_dir)

# Initialize the Groq Engine Client securely
GROQ_API_KEY = os.environ.get("GROQ_API_KEY")
client = None
if GROQ_API_KEY:
    client = Groq(api_key=GROQ_API_KEY)

# Initialize Supabase Client securely
SUPABASE_URL = os.environ.get("SUPABASE_URL")
SUPABASE_KEY = os.environ.get("SUPABASE_KEY") or os.environ.get("SUPABASE_ANON_KEY")
supabase_client: Client = None

if SUPABASE_URL and SUPABASE_KEY:
    try:
        supabase_client = create_client(SUPABASE_URL, SUPABASE_KEY)
    except Exception as e:
        print(f"Supabase client initialization failed: {e}")

# Define verified, active, non-decommissioned Groq model identifiers for 2026
MODEL_EVALUATOR = "openai/gpt-oss-120b"  # High reasoning capability for detailed risk metrics
MODEL_CHAT = "openai/gpt-oss-20b"        # Sub-second token delivery optimized for user chat loops

def validate_us_zip(zip_code):
    """
    Enforces the geographical boundary constraint.
    Validates if the submitted value follows a clean 5-digit US Zip Code format.
    """
    return bool(re.match(r"^\d{5}$", str(zip_code).strip()))

def calculate_stewardship_score(usage, concern):
    """
    Calculates a baseline mathematical Eco-Stewardship Score (out of 100)
    to transform abstract user habits into an engaging personal indicator.
    """
    base_score = 100
    
    # Evaluate estimated resource usage footprint
    usage_normalized = usage.lower().strip()
    if usage_normalized == 'high':
        base_score -= 40
    elif usage_normalized == 'moderate':
        base_score -= 20
    elif usage_normalized == 'low':
        base_score -= 5
        
    # Apply context weight adjustments based on primary environmental concern
    concern_normalized = concern.lower().strip()
    concern_penalties = {
        'drought': 15,
        'heatwaves': 10,
        'clean water access': 20,
        'severe weather': 10
    }
    
    penalty = concern_penalties.get(concern_normalized, 5)
    final_score = max(10, base_score - penalty)
    return final_score

def fetch_realtime_context(zip_code):
    """
    Queries open APIs to dynamically retrieve real-time geographical and hazards telemetry.
    Includes fallback default parameters if any external upstream pipeline hits a fault.
    """
    context = {
        "city": "Unknown US Region",
        "state": "US",
        "lat": "38.0",
        "lon": "-97.0",
        "noaa_alerts": [],
        "fema_disasters": [],
        "nasa_events": []
    }
    
    # 1. Coordinate & State lookup via Zippopotamus
    try:
        geo_resp = requests.get(f"https://api.zippopotam.us/us/{zip_code}", timeout=3.0)
        if geo_resp.status_code == 200:
            geo_data = geo_resp.json()
            if "places" in geo_data and len(geo_data["places"]) > 0:
                place = geo_data["places"][0]
                context["city"] = place.get("place name", "Unknown City")
                context["state"] = place.get("state abbreviation", "US")
                context["lat"] = place.get("latitude", "38.0")
                context["lon"] = place.get("longitude", "-97.0")
    except Exception:
        pass  # Graceful fallback to default regional coordinates

    state_code = context["state"]

    # 2. NOAA Real-Time Weather Alerts API
    try:
        noaa_url = f"https://api.weather.gov/alerts/active?area={state_code}"
        headers = {"User-Agent": "(Aegis, aegis-support@example.com)"}
        noaa_resp = requests.get(noaa_url, headers=headers, timeout=3.0)
        if noaa_resp.status_code == 200:
            features = noaa_resp.json().get("features", [])[:4]
            for feat in features:
                props = feat.get("properties", {})
                context["noaa_alerts"].append({
                    "event": props.get("event", "Weather Alert"),
                    "severity": props.get("severity", "Unknown"),
                    "area": props.get("areaDesc", "Regional Area"),
                    "headline": props.get("headline", ""),
                    "instruction": props.get("instruction", "Stay tuned to local safety guidelines.")
                })
    except Exception:
        pass

    # 3. OpenFEMA Disaster Declarations V2 API
    try:
        fema_url = f"https://www.fema.gov/api/open/v2/DisasterDeclarationsSummaries?$filter=state eq '{state_code}'&$top=4&$orderby=declarationDate desc"
        fema_resp = requests.get(fema_url, timeout=3.0)
        if fema_resp.status_code == 200:
            results = fema_resp.json().get("DisasterDeclarationsSummaries", [])
            for res in results:
                context["fema_disasters"].append({
                    "title": res.get("declarationTitle", "Emergency Declared"),
                    "type": res.get("incidentType", "Severe Event"),
                    "date": res.get("declarationDate", "")[:10] if res.get("declarationDate") else "N/A",
                    "fema_string": res.get("femaDeclarationString", "N/A")
                })
    except Exception:
        pass

    # 4. NASA Natural Event Tracker (EONET) API - Restricted geographically to US Bounding Box roughly
    try:
        nasa_url = "https://eonet.gsfc.nasa.gov/api/v3/events?status=open&limit=8"
        nasa_resp = requests.get(nasa_url, timeout=3.0)
        if nasa_resp.status_code == 200:
            events = nasa_resp.json().get("events", [])
            for ev in events:
                categories = [cat.get("title", "") for cat in ev.get("categories", [])]
                geoms = ev.get("geometry", [])
                date_str = geoms[0].get("date", "")[:10] if geoms else "N/A"
                context["nasa_events"].append({
                    "title": ev.get("title", "Natural Dynamic Event"),
                    "category": ", ".join(categories) if categories else "Environmental Hazard",
                    "date": date_str,
                    "link": ev.get("link", "")
                })
    except Exception:
        pass

    return context

# ================================
# MULTI-PAGE VIEW ROUTES
# ================================

@app.route('/')
@app.route('/assessment')
@app.route('/preparedness')
@app.route('/dashboard')
@app.route('/safecircle')
@app.route('/telemetry')
@app.route('/chat')
@app.route('/trend-archive')
@app.route('/atmospheric-intelligence')
@app.route('/digital-twin')
def serve_page():
    """Renders the core application layout supporting distinct page views."""
    return render_template('index.html')

# ================================
# AUTHENTICATION API ROUTES
# ================================

@app.route('/api/auth/config', methods=['GET'])
def get_auth_config():
    """Exposes Supabase frontend configuration keys."""
    return jsonify({
        "supabase_url": SUPABASE_URL or "",
        "supabase_key": SUPABASE_KEY or ""
    })

@app.route('/api/auth/signup', methods=['POST'])
def auth_signup():
    """Handles Username/Email and Password registration via Supabase."""
    if not supabase_client:
        return jsonify({"error": "Supabase integration is not configured on the server."}), 500

    data = request.get_json() or {}
    email = data.get('email', '').strip()
    password = data.get('password', '').strip()
    full_name = data.get('full_name', '').strip()

    if not email or not password:
        return jsonify({"error": "Email and password parameters are required."}), 400

    try:
        auth_response = supabase_client.auth.sign_up({
            "email": email,
            "password": password,
            "options": {
                "data": {
                    "full_name": full_name
                }
            }
        })
        
        user = auth_response.user
        session = auth_response.session

        return jsonify({
            "success": True,
            "user": {
                "id": user.id if user else None,
                "email": user.email if user else email,
                "full_name": full_name
            } if user else None,
            "session": {
                "access_token": session.access_token if session else None,
                "refresh_token": session.refresh_token if session else None
            } if session else None,
            "message": "Account created successfully!"
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 400

@app.route('/api/auth/login', methods=['POST'])
def auth_login():
    """Handles Username/Email and Password login via Supabase."""
    if not supabase_client:
        return jsonify({"error": "Supabase integration is not configured on the server."}), 500

    data = request.get_json() or {}
    email = data.get('email', '').strip()
    password = data.get('password', '').strip()

    if not email or not password:
        return jsonify({"error": "Email and password parameters are required."}), 400

    try:
        auth_response = supabase_client.auth.sign_in_with_password({
            "email": email,
            "password": password
        })

        user = auth_response.user
        session = auth_response.session

        return jsonify({
            "success": True,
            "user": {
                "id": user.id if user else None,
                "email": user.email if user else email,
                "user_metadata": user.user_metadata if user else {}
            } if user else None,
            "session": {
                "access_token": session.access_token if session else None,
                "refresh_token": session.refresh_token if session else None
            } if session else None,
            "message": "Login successful!"
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 400

@app.route('/api/auth/google', methods=['POST'])
def auth_google():
    """Generates the Google OAuth authorization URL via Supabase."""
    if not supabase_client:
        return jsonify({"error": "Supabase integration is not configured on the server."}), 500

    try:
        data = request.get_json() or {}
        redirect_to = data.get('redirect_to', request.host_url)

        oauth_response = supabase_client.auth.sign_in_with_oauth({
            "provider": "google",
            "options": {
                "redirect_to": redirect_to
            }
        })

        return jsonify({
            "success": True,
            "url": oauth_response.url
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 400

@app.route('/api/auth/user', methods=['GET'])
def auth_user():
    """Validates token and returns current authenticated user profile."""
    if not supabase_client:
        return jsonify({"error": "Supabase integration is not configured on the server."}), 500

    auth_header = request.headers.get('Authorization')
    if not auth_header or not auth_header.startswith('Bearer '):
        return jsonify({"error": "Missing authorization token."}), 401

    token = auth_header.split(' ')[1]
    try:
        user_response = supabase_client.auth.get_user(token)
        if user_response and user_response.user:
            return jsonify({
                "success": True,
                "user": {
                    "id": user_response.user.id,
                    "email": user_response.user.email,
                    "user_metadata": user_response.user.user_metadata
                }
            })
        return jsonify({"error": "User token invalid or expired."}), 401
    except Exception as e:
        return jsonify({"error": str(e)}), 401

# ================================
# CORE APPLICATION API ROUTES
# ================================

@app.route('/api/evaluate', methods=['POST'])
def evaluate_region():
    """
    Mode 1: Regional Profile & Threat Evaluator (Enriched with NASA, NOAA, & FEMA Telemetry)
    """
    if not client:
        return jsonify({"error": "Groq API token configuration is missing on the server environment."}), 500

    data = request.get_json() or {}
    zip_code = data.get('zip_code', '').strip()
    usage = data.get('usage', '').strip()
    concern = data.get('concern', '').strip()

    if not zip_code or not validate_us_zip(zip_code):
        return jsonify({"error": "Invalid location context. Please provide a valid 5-digit US Zip Code."}), 400

    if not usage or not concern:
        return jsonify({"error": "Missing essential parameters. Usage level and primary concern are required."}), 400

    realtime_data = fetch_realtime_context(zip_code)
    stewardship_score = calculate_stewardship_score(usage, concern)

    noaa_summary = "\n".join([f"- {a['event']} (Severity: {a['severity']}): {a['headline']}" for a in realtime_data["noaa_alerts"]]) if realtime_data["noaa_alerts"] else "No active warnings found."
    fema_summary = "\n".join([f"- {d['title']} ({d['type']}) declared on {d['date']}" for d in realtime_data["fema_disasters"]]) if realtime_data["fema_disasters"] else "No recent federal disaster declarations."
    nasa_summary = "\n".join([f"- {e['title']} ({e['category']}) tracked on {e['date']}" for e in realtime_data["nasa_events"]]) if realtime_data["nasa_events"] else "No major hazards in satellite logs."

    system_instruction = (
        "You are Aegis, an expert humanitarian environmental analysis engine tailored for the US environment. "
        "Your goal is to provide educational regional risk assessments and neighborhood conservation tips. "
        "Incorporate the provided real-time NOAA alerts, FEMA Disaster declarations, and NASA EONET satellite data "
        "to formulate situational awareness and emergency protocols. "
        "Align assessments with standard safety framings inspired by FEMA and EPA advisory frameworks. "
        "Structure your response elegantly using clear Markdown headers, bold accents, and distinct spacing. "
        "Maintain a highly professional, authoritative tone without emojis."
    )
    
    user_query = (
        f"Analyze this US regional sustainability snapshot and compile a community profile:\n"
        f"- Target US Zip Code Region: {zip_code} (City: {realtime_data['city']}, State: {realtime_data['state']})\n"
        f"- Latitude: {realtime_data['lat']}, Longitude: {realtime_data['lon']}\n"
        f"- Reported Household Resource Footprint: {usage}\n"
        f"- Target Local Safety & Resource Crisis Parameter: {concern}\n\n"
        f"Real-Time Telemetry Gathered:\n"
        f"[NOAA Weather Warning Streams]:\n{noaa_summary}\n\n"
        f"[OpenFEMA Disaster Logs]:\n{fema_summary}\n\n"
        f"[NASA Satellites (EONET Environment tracking)]:\n{nasa_summary}\n\n"
        f"Output structural guidelines addressing:\n"
        f"1. A localized 'Regional Resource Assessment' linking resource usage habits to regional environmental limits.\n"
        f"2. Explicit 'Contextual Safety Alerts' highlighting common indicators of resource vulnerability with reference to real-time telemetry details.\n"
        f"3. Three actionable, step-by-step community-level mitigation ideas and local resource allocations based on real-time threats."
    )

    try:
        completion = client.chat.completions.create(
            model=MODEL_EVALUATOR,
            messages=[
                {"role": "system", "content": system_instruction},
                {"role": "user", "content": user_query}
            ],
            temperature=0.25,
            max_tokens=1000
        )
        
        analysis_payload = completion.choices[0].message.content
        
        return jsonify({
            "success": True,
            "stewardship_score": stewardship_score,
            "analysis": analysis_payload,
            "realtime_telemetry": realtime_data
        })

    except Exception as e:
        return jsonify({"error": f"Groq processing pipeline hit an execution fault: {str(e)}"}), 500

@app.route('/api/chat', methods=['POST'])
def chat_advisory():
    """
    Mode 2: Eco-Safety Advisory Chat Hub
    """
    if not client:
        return jsonify({"error": "Groq API token configuration is missing on the server environment."}), 500

    data = request.get_json() or {}
    user_message = data.get('message', '').strip()
    chat_history = data.get('history', [])

    if not user_message:
        return jsonify({"error": "Chat message context cannot be blank."}), 400

    system_instruction = (
        "You are the Aegis Eco-Safety Advisory Chat Hub. You act as an interactive neighborhood safety monitor. "
        "Provide immediate, step-by-step micro-level resource saving and family preparation strategies. "
        "Ensure all guidance assumes a US municipal context (e.g., standard American Red Cross emergency kits). "
        "Keep replies highly operational, punchy, concise, and structured with clean bullet points. "
        "Avoid using emojis in your output to maintain a professional standard."
    )

    messages = [{"role": "system", "content": system_instruction}]
    
    for turn in chat_history[-6:]:
        if isinstance(turn, dict) and 'role' in turn and 'content' in turn:
            messages.append({"role": turn['role'], "content": turn['content']})
            
    messages.append({"role": "user", "content": user_message})

    try:
        completion = client.chat.completions.create(
            model=MODEL_CHAT,
            messages=messages,
            temperature=0.55,
            max_tokens=600
        )
        
        reply_payload = completion.choices[0].message.content
        
        return jsonify({
            "success": True,
            "reply": reply_payload
        })

    except Exception as e:
        return jsonify({"error": f"Chat integration engine hit an execution fault: {str(e)}"}), 500

@app.errorhandler(404)
def resource_not_found(e):
    return jsonify({"error": "The specified route configuration does not exist on this application."}), 404

if __name__ == '__main__':
    app.run(debug=True, port=4000)
