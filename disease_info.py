def get_disease_info(class_name):
    """
    Get disease information based on the predicted class name
    """
    
    # Disease information database
    disease_db = {
        # POTATO DISEASES
        'Potato___Early_blight': {
            'plant': 'Potato',
            'status': 'Diseased',
            'disease': 'Early Blight',
            'cause': 'Caused by the fungus Alternaria solani. Thrives in warm, humid conditions with temperatures between 24-29°C. Spreads through infected plant debris, contaminated seeds, and wind-borne spores.',
            'treatment': '1. Apply fungicides containing chlorothalonil, mancozeb, or copper-based compounds\n2. Remove and destroy infected leaves\n3. Spray fungicides every 7-10 days during humid weather\n4. Use bio-fungicides like Bacillus subtilis',
            'prevention': '1. Use disease-resistant potato varieties\n2. Practice crop rotation (avoid planting potatoes in same location for 2-3 years)\n3. Ensure proper spacing for air circulation\n4. Water plants at the base to keep foliage dry\n5. Remove plant debris after harvest\n6. Use certified disease-free seed potatoes'
        },
        'Potato___Late_blight': {
            'plant': 'Potato',
            'status': 'Diseased',
            'disease': 'Late Blight',
            'cause': 'Caused by the oomycete Phytophthora infestans. This is the same pathogen that caused the Irish Potato Famine. Thrives in cool, wet weather with temperatures 10-25°C and high humidity. Spreads rapidly through airborne spores.',
            'treatment': '1. Apply protective fungicides immediately (mancozeb, chlorothalonil, or metalaxyl)\n2. Remove and destroy all infected plants and tubers\n3. Apply copper-based fungicides as preventive measure\n4. Spray every 5-7 days during wet weather\n5. Burn infected plant material (do not compost)',
            'prevention': '1. Plant resistant varieties like Defender or Sarpo Mira\n2. Use certified disease-free seed potatoes\n3. Avoid overhead irrigation\n4. Provide adequate plant spacing\n5. Hill up soil around plants to protect tubers\n6. Monitor weather and apply preventive fungicides before rain\n7. Remove volunteer potato plants\n8. Harvest in dry weather'
        },
        'Potato___healthy': {
            'plant': 'Potato',
            'status': 'Healthy',
            'disease': 'None',
            'cause': 'N/A',
            'treatment': 'N/A',
            'prevention': 'Continue good agricultural practices: proper watering, adequate nutrition, crop rotation, and regular monitoring for pests and diseases.'
        },
        
        # TOMATO DISEASES (if in dataset)
        'Tomato___Early_blight': {
            'plant': 'Tomato',
            'status': 'Diseased',
            'disease': 'Early Blight',
            'cause': 'Caused by Alternaria solani fungus. Common in warm, humid conditions.',
            'treatment': 'Remove infected leaves, apply fungicides containing chlorothalonil or copper compounds.',
            'prevention': 'Use resistant varieties, practice crop rotation, mulch around plants, water at base.'
        },
        
        # GRAPE DISEASES
        'Grape___Black_rot': {
            'plant': 'Grape',
            'status': 'Diseased',
            'disease': 'Black Rot',
            'cause': 'Caused by the fungus Guignardia bidwellii. Thrives in warm, humid conditions with temperatures 20-27°C. Overwinters in infected fruit mummies and canes. Spreads through rain splash and wind.',
            'treatment': '1. Remove and destroy all infected fruit, leaves, and mummified berries\n2. Apply fungicides: myclobutanil, captan, or mancozeb\n3. Prune infected canes during dormant season\n4. Apply fungicides from bud break through fruit set\n5. Spray every 7-14 days during wet weather',
            'prevention': '1. Remove mummified berries from vines and ground\n2. Prune vines to improve air circulation\n3. Keep vineyard floor clean of debris\n4. Apply dormant spray of lime sulfur\n5. Use resistant grape varieties when possible\n6. Avoid overhead irrigation\n7. Train vines properly for sunlight exposure'
        },
        'Grape___Esca_(Black_Measles)': {
            'plant': 'Grape',
            'status': 'Diseased',
            'disease': 'Esca (Black Measles)',
            'cause': 'Complex disease caused by multiple fungi including Phaeomoniella chlamydospora and Phaeoacremonium species. Enters through pruning wounds. Associated with old vines and stress conditions. No cure available once established.',
            'treatment': '1. Remove severely infected vines\n2. Prune out infected wood during dormant season\n3. Apply protective fungicides to pruning wounds\n4. Maintain vine health through proper nutrition and irrigation\n5. Some growers report success with trunk renewal/surgery',
            'prevention': '1. Make pruning cuts during dry weather\n2. Apply pruning wound protectants immediately after cutting\n3. Use clean, disinfected pruning tools\n4. Train new cordons from healthy wood\n5. Avoid stress through proper irrigation and fertilization\n6. Remove infected vines to prevent spread\n7. Consider double Guyot or replacement cane training systems'
        },
        'Grape___Leaf_blight': {
            'plant': 'Grape',
            'status': 'Diseased',
            'disease': 'Leaf Blight (Isariopsis Leaf Spot)',
            'cause': 'Caused by the fungus Pseudocercospora vitis (formerly Isariopsis clavispora). Thrives in warm, humid conditions. Spreads through rain splash and high humidity.',
            'treatment': '1. Remove and destroy infected leaves\n2. Apply copper-based fungicides\n3. Use systemic fungicides like tebuconazole or difenoconazole\n4. Spray every 10-14 days during disease-favorable conditions\n5. Improve canopy air circulation through pruning',
            'prevention': '1. Practice good vineyard sanitation\n2. Remove fallen leaves from vineyard floor\n3. Ensure proper vine spacing and trellising\n4. Prune for adequate air circulation\n5. Avoid excessive nitrogen fertilization\n6. Apply preventive fungicide sprays during humid periods\n7. Remove lower leaves to reduce humidity in canopy'
        },
        'Grape___healthy': {
            'plant': 'Grape',
            'status': 'Healthy',
            'disease': 'None',
            'cause': 'N/A',
            'treatment': 'N/A',
            'prevention': 'Maintain good vineyard practices: proper pruning, adequate spacing, balanced fertilization, and regular disease monitoring.'
        },
        
        # CORN DISEASES
        'Corn___Cercospora_leaf_spot': {
            'plant': 'Corn',
            'status': 'Diseased',
            'disease': 'Cercospora Leaf Spot',
            'cause': 'Caused by fungi Cercospora zeae-maydis and C. zeina. Thrives in warm (22-30°C), humid conditions with prolonged leaf wetness. Spreads through infected crop residue and airborne spores. More severe under no-till systems.',
            'treatment': '1. Apply foliar fungicides: azoxystrobin, pyraclostrobin, or triazole fungicides\n2. Spray at first sign of disease or preventively at V8-VT growth stage\n3. Use systemic fungicides for better control\n4. May require multiple applications in severe conditions\n5. Fungicides most effective when applied before disease is widespread',
            'prevention': '1. Plant resistant hybrids (most effective prevention)\n2. Practice crop rotation (2-3 years)\n3. Till under corn residue to reduce inoculum\n4. Avoid planting corn after corn\n5. Ensure adequate plant spacing\n6. Apply balanced fertilization\n7. Scout fields regularly during late vegetative growth\n8. Plant early to avoid high-risk weather periods'
        },
        'Corn___Common_rust': {
            'plant': 'Corn',
            'status': 'Diseased',
            'disease': 'Common Rust',
            'cause': 'Caused by the fungus Puccinia sorghi. Thrives in cool (16-23°C), humid conditions with heavy dew. Spores are wind-dispersed over long distances. More common in temperate regions and at higher elevations.',
            'treatment': '1. Apply foliar fungicides if disease appears before tasseling: triazole or strobilurin fungicides\n2. Treatment often not economical after pollination\n3. Monitor disease severity and weather conditions\n4. Fungicide application most beneficial during grain fill if severe\n5. Products: propiconazole, tebuconazole, azoxystrobin',
            'prevention': '1. Plant resistant hybrids (check resistance ratings)\n2. Scout fields from early whorl stage through grain fill\n3. Plant corn at recommended times\n4. Avoid water stress through proper irrigation\n5. Ensure balanced soil fertility\n6. Apply preventive fungicides in high-risk areas\n7. Different resistance genes available for different regions'
        },
        'Corn___Northern_Leaf_Blight': {
            'plant': 'Corn',
            'status': 'Diseased',
            'disease': 'Northern Leaf Blight',
            'cause': 'Caused by the fungus Exserohilum turcicum (Setosphaeria turcica). Thrives in moderate temperatures (18-27°C) with high humidity and prolonged leaf wetness. Overwinters in corn residue. Spreads through airborne spores and rain splash.',
            'treatment': '1. Apply foliar fungicides at first disease symptoms\n2. Products: azoxystrobin, pyraclostrobin, propiconazole, or combinations\n3. Spray between V8 and R1 growth stages for best results\n4. May need repeat application 14-21 days later\n5. Most effective when applied before disease reaches mid-canopy',
            'prevention': '1. Plant resistant hybrids (most economical approach)\n2. Practice crop rotation (minimum 1 year)\n3. Till under corn debris to reduce inoculum\n4. Avoid susceptible hybrids in high-risk areas\n5. Ensure adequate plant nutrition, especially potassium\n6. Avoid excessive plant density\n7. Scout fields regularly from late vegetative stages\n8. Consider preventive fungicide in high-risk environments'
        },
        'Corn___healthy': {
            'plant': 'Corn',
            'status': 'Healthy',
            'disease': 'None',
            'cause': 'N/A',
            'treatment': 'N/A',
            'prevention': 'Continue best practices: use quality seeds, maintain proper plant density, ensure adequate fertilization, practice crop rotation, and monitor for pests and diseases.'
        },
        
        # APPLE DISEASES
        'Apple___Apple_scab': {
            'plant': 'Apple',
            'status': 'Diseased',
            'disease': 'Apple Scab',
            'cause': 'Caused by the fungus Venturia inaequalis. Thrives in cool (13-24°C), wet spring weather. Overwinters in fallen infected leaves. Primary infection occurs during spring rains. One of the most serious apple diseases worldwide.',
            'treatment': '1. Apply fungicides during infection periods: captan, myclobutanil, or dodine\n2. Start sprays at green tip and continue through petal fall\n3. Spray every 7-10 days during wet weather\n4. Remove scabbed fruit to reduce inoculum\n5. Rake and destroy fallen leaves\n6. Use systemic fungicides for better control',
            'prevention': '1. Plant scab-resistant varieties (Liberty, Enterprise, Freedom)\n2. Rake and remove fallen leaves in autumn\n3. Prune trees for good air circulation\n4. Apply urea to fallen leaves to accelerate decomposition\n5. Avoid overhead irrigation\n6. Space trees adequately\n7. Apply dormant lime-sulfur spray\n8. Monitor weather for infection periods (rain + temperature)'
        },
        'Apple___Black_rot': {
            'plant': 'Apple',
            'status': 'Diseased',
            'disease': 'Black Rot',
            'cause': 'Caused by the fungus Botryosphaeria obtusa. Enters through wounds and causes fruit rot, leaf spot ("frogeye leaf spot"), and limb cankers. Thrives in warm, humid conditions. Overwinters in cankers and infected fruit mummies.',
            'treatment': '1. Remove and destroy infected fruit (mummies) and dead wood\n2. Prune out cankered branches during dormant season\n3. Apply fungicides: captan, thiophanate-methyl, or ziram\n4. Spray from pink bud through second cover spray\n5. Focus on sanitation as primary control\n6. Paint large pruning wounds with wound dressing',
            'prevention': '1. Remove all mummified fruit from tree and ground\n2. Prune out dead wood and cankers during winter\n3. Improve tree vigor through proper fertilization and watering\n4. Maintain good air circulation through pruning\n5. Avoid tree stress and injury\n6. Remove wild apple trees near orchard\n7. Practice good orchard sanitation\n8. Make clean pruning cuts that heal properly'
        },
        'Apple___Cedar_apple_rust': {
            'plant': 'Apple',
            'status': 'Diseased',
            'disease': 'Cedar Apple Rust',
            'cause': 'Caused by the fungus Gymnosporangium juniperi-virginianae. Requires both apple/crabapple and eastern red cedar (juniper) to complete life cycle. Spores from cedar galls infect apple in spring during wet weather. Most active in cool, wet springs.',
            'treatment': '1. Apply protective fungicides during spring infection periods\n2. Products: myclobutanil, mancozeb, or sulfur\n3. Start sprays at pink bud stage\n4. Continue every 7-10 days until dry weather\n5. Treatment is preventive - cannot cure existing infections\n6. Remove nearby cedar/juniper trees if possible',
            'prevention': '1. Plant rust-resistant apple varieties (most effective)\n2. Remove nearby eastern red cedar trees within 2 miles (if practical)\n3. Apply preventive fungicides from pink through petal fall\n4. Prune apple trees for good air circulation\n5. Remove galls from cedar trees in winter (small scale)\n6. Select site away from cedars when planting new orchards\n7. Monitor weather for infection periods in spring'
        },
        'Apple___healthy': {
            'plant': 'Apple',
            'status': 'Healthy',
            'disease': 'None',
            'cause': 'N/A',
            'treatment': 'N/A',
            'prevention': 'Maintain orchard health: proper pruning, balanced nutrition, adequate watering, good sanitation, and regular pest and disease monitoring.'
        }
    }
    
    # Return disease info or default message
    if class_name in disease_db:
        return disease_db[class_name]
    else:
        # Parse class name for unknown diseases
        parts = class_name.replace('_', ' ').split('___')
        plant = parts[0] if len(parts) > 0 else 'Unknown'
        disease = parts[1] if len(parts) > 1 else 'Unknown'
        
        return {
            'plant': plant,
            'status': 'Diseased' if 'healthy' not in class_name.lower() else 'Healthy',
            'disease': disease,
            'cause': 'Information not available in database.',
            'treatment': 'Please consult with a local agricultural extension office or plant pathologist.',
            'prevention': 'Follow general good agricultural practices and disease management guidelines.'
        }