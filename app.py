import json
import streamlit as st
from pathlib import Path
from huggingface_hub import hf_hub_download

st.set_page_config(page_title="Annotatie Platform", layout="wide")

# **1. Originele dataset laden (train + test) vanuit Hugging Face**
@st.cache_data
def load_original_dataset():
    try:
        # Download dataset-bestanden van Hugging Face hub
        train_path = hf_hub_download(
            repo_id="nvidia/CantTalkAboutThis-Topic-Control-Dataset",
            filename="canttalkaboutthis_topic_control_mixtral.jsonl", repo_type="dataset"
        )
        test_path = hf_hub_download(
            repo_id="nvidia/CantTalkAboutThis-Topic-Control-Dataset",
            filename="canttalkaboutthis_topic_control_human_test_set.jsonl", repo_type="dataset"
        )
        records = []
        # Lees train
        with open(train_path, 'r', encoding='utf-8') as f:
            for line in f:
                row = json.loads(line)
                row['split'] = 'train'  # extra kolom voor split
                records.append(row)
        # Lees test
        with open(test_path, 'r', encoding='utf-8') as f:
            for line in f:
                row = json.loads(line)
                row['split'] = 'test'
                records.append(row)
        return records
    except Exception as e:
        st.error(f"Kon originele dataset niet laden: {e}")
        return []

original_data = load_original_dataset()

# Lijst van unieke domeinen in originele dataset
domeinen = sorted(set([r.get("domain","") for r in original_data if r.get("domain")]))
splits = ["train", "test"]

# Sessiestate voor nieuwe/gewijzigde entries
if "entries" not in st.session_state:
    st.session_state["entries"] = []

# **Navigatie sidebar**
st.sidebar.title("Navigatie")
pagina = st.sidebar.radio("Pagina", ["Bladeren", "Toevoegen", "Exporteren"])

# --- Paginacontent ---

if pagina == "Bladeren":
    st.title("Originele dataset doorlopen")
    st.info("Gebruik de filters om de dataset te doorzoeken.")
    col1, col2 = st.columns([2,1])
    with col2:
        # Filteropties
        gekozen_domein = st.selectbox("Filter op domein", [""] + domeinen)
        chosen_split = st.selectbox("Filter op split", [""] + splits)
    # Filter de originele data
    df = []
    for rec in original_data:
        if gekozen_domein and rec.get("domain") != gekozen_domein:
            continue
        if chosen_split and rec.get("split") != chosen_split:
            continue
        # Voorbeeld van het gesprek tonen (kort)
        preview = ""
        convo = rec.get("conversation", [])
        if convo:
            # Toon laatste user-instructie of anders laatste boodschap
            for turn in reversed(convo):
                if turn.get("role") == "user":
                    preview = turn.get("content", "")
                    break
            if not preview and convo:
                preview = convo[-1].get("content", "")
        df.append({
            "Domein": rec.get("domain",""),
            "Situatie": rec.get("scenario",""),
            "Voorbeeld": preview[:50] + ("..." if len(preview)>50 else ""),
            "Split": rec.get("split","")
        })
    st.dataframe(df, use_container_width=True)

elif pagina == "Toevoegen":
    st.title("Nieuwe entry toevoegen / bestaande bewerken")
    modus = st.radio("Modus", ["Nieuwe entry", "Bestaande entry bewerken"])
    # Selectie bestaande entry
    if modus == "Bestaande entry bewerken":
        col1, col2 = st.columns([1,1])
        with col1:
            domein_sel = st.selectbox("Selecteer domein", [""] + domeinen)
        with col2:
            scenario_options = []
            if domein_sel:
                scenario_options = sorted(set(
                    [r["scenario"] for r in original_data if r.get("domain")==domein_sel]
                ))
            gekozen_scenario = st.selectbox("Selecteer situatie", [""] + scenario_options)
        entry = None
        if domein_sel and gekozen_scenario:
            matches = [r for r in original_data if r.get("domain")==domein_sel and r.get("scenario")==gekozen_scenario]
            entry = matches[0] if matches else None
    else:
        entry = None

    # Default-waarden
    if entry:
        domain_val = entry.get("domain","")
        scenario_val = entry.get("scenario","")
        instr = entry.get("system_instruction","")
        conv_list = entry.get("conversation",[])
        distractors_list = entry.get("distractors",[])
        split_val = entry.get("split","train")
    else:
        domain_val = ""
        scenario_val = ""
        instr = ""
        conv_list = []
        distractors_list = []
        split_val = "train"

    # **Invoervelden**
    col1, col2 = st.columns([1,1])
    with col1:
        domein_in = st.selectbox("Domein", [""] + domeinen, index=0)
        nieuwe_domein = st.text_input("Nieuw domein (indien niet in lijst)", value="", placeholder="Optioneel")
    with col2:
        situatie_in = st.text_input("Situatie (scenario)", value=scenario_val)
        nieuwe_situatie = st.text_input("Nieuwe situatie (indien gewenst)", value="", placeholder="Optioneel")
    systeem_instr = st.text_area("Systeeminstructie", value=instr, height=100)

    st.markdown("**Gesprek zonder afleiders:** Voer het gesprek in. Gebruik per regel `User: ...` of `Assistant: ...`.")
    gesprek_txt = st.text_area("Gesprek", value="", height=150,
                                placeholder="Bijv:\nUser: Hallo, hoe kan ik je helpen?\nAssistant: ...")
    st.markdown("**Afleiders (distractors):** Voer de afleidende gebruikersvragen in (max 5).")
    afleidingen = []
    # Splits de 5 invoervelden in twee kolommen
    cols = st.columns(2)
    for i in range(5):
        with cols[i%2]:
            af = st.text_input(f"Afleider {i+1}", 
                                value=(distractors_list[i]["distractor"] if i < len(distractors_list) else ""))
            afleidingen.append(af)
    split_keuze = st.radio("Markeer als 'train' of 'test'", ["train", "test"], index=0 if split_val=="train" else 1)

    if st.button("Opslaan"):
        # Bepaal domein en situatie (nieuw of bestaand)
        domein_final = (nieuwe_domein.strip() or domein_in or "").strip()
        scenario_final = (nieuwe_situatie.strip() or situatie_in or "").strip()
        if not domein_final or not scenario_final:
            st.error("Vul zowel domein als situatie in.")
        else:
            # Conversatie parseren
            gesprek_regels = []
            for line in gesprek_txt.splitlines():
                if ":" in line:
                    role, msg = line.split(":", 1)
                    gesprek_regels.append({"role": role.strip(), "content": msg.strip()})
            # Als leeg gelaten, gebruik uit bestaande entry
            if not gesprek_regels:
                gesprek_regels = conv_list
            # Vind laatste assistant-bericht voor 'bot_turn'
            laatste_assistent = ""
            for turn in reversed(gesprek_regels):
                if turn.get("role","").lower() == "assistant":
                    laatste_assistent = turn.get("content","")
                    break
            # Bouw list van distractors
            nieuwe_afleiders = []
            for af in afleidingen:
                af = af.strip()
                if af:
                    nieuwe_afleiders.append({"bot_turn": laatste_assistent, "distractor": af})
            # Bouw gesprek met afleiders sequentieel
            conv_met_afl = []
            base_conv = gesprek_regels.copy()
            conv_met_afl.append(base_conv.copy())
            for af in nieuwe_afleiders:
                base_conv = base_conv.copy()
                if laatste_assistent:
                    base_conv.append({"role": "assistant", "content": laatste_assistent})
                base_conv.append({"role": "user", "content": af["distractor"]})
                conv_met_afl.append(base_conv.copy())
            # Gegevens entry samenstellen
            entry_data = {
                "domain": domein_final,
                "scenario": scenario_final,
                "system_instruction": systeem_instr,
                "conversation": gesprek_regels,
                "distractors": nieuwe_afleiders,
                "conversation_with_distractors": conv_met_afl,
                "split": split_keuze
            }
            st.session_state["entries"].append(entry_data)
            st.success("Entry is opgeslagen.")
            st.json(entry_data)  # Toon de opgeslagen data

elif pagina == "Exporteren":
    st.title("Exporteren van de dataset")
    st.info("Exporteer (gecombineerde) dataset als JSONL of CSV.")
    includ_orig = st.checkbox("Inclusief originele dataset (train+test)", value=True)
    entries = st.session_state.get("entries", [])
    merged = []
    if includ_orig:
        merged = list(original_data)  # kopie originele data
    for e in entries:
        merged.append(e)
    if not merged:
        st.info("Nog geen data om te exporteren.")
    else:
        # JSONL export
        jsonl_data = "\n".join([json.dumps(rec, ensure_ascii=False) for rec in merged])
        st.download_button("Download JSONL", data=jsonl_data, 
                           file_name="dataset_export.jsonl", mime="application/json")
        # CSV export (flatten nested velden als JSON-strings)
        import pandas as pd
        df_export = pd.json_normalize(merged)
        df_export.fillna("", inplace=True)
        csv_data = df_export.to_csv(index=False).encode('utf-8')
        st.download_button("Download CSV", data=csv_data, 
                           file_name="dataset_export.csv", mime="text/csv")
