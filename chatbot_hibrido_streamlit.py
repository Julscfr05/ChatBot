
import ast
import csv
import json
import operator as op
import re
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import datetime
from difflib import SequenceMatcher, get_close_matches
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import streamlit as st


APP_TITLE = "Chatbot híbrido PRO"
DEFAULT_THRESHOLD = 0.58
KNOWLEDGE_FOLDER = Path("bases")
LEARNING_FILE = Path("aprendizaje_usuario.json")
LOG_FILE = Path("consultas.csv")
CONVERSATION_EXPORT_FILE = "conversacion_chatbot.txt"
MAX_PRIMARY_BASES = 5
ADMIN_PASSWORD = "Admin2026!"  # cámbiala si deseas


# =========================
# Texto y normalización
# =========================
STOPWORDS = {
    "que", "de", "la", "el", "los", "las", "un", "una", "unos", "unas",
    "es", "son", "fue", "fueron", "quien", "quién", "cual", "cuál",
    "cuales", "cuáles", "como", "cómo", "cuando", "cuándo", "donde",
    "dónde", "por", "para", "del", "al", "en", "y", "o", "se", "su",
    "sus", "mi", "tu", "te", "me", "lo", "le", "les", "con", "with", "what"
}

MATH_WORDS = [
    (r"\bmas\b", "+"),
    (r"\bmás\b", "+"),
    (r"\bmenos\b", "-"),
    (r"\bpor\b", "*"),
    (r"\bentre\b", "/"),
    (r"\bdividido entre\b", "/"),
    (r"\bdividido por\b", "/"),
]

_ALLOWED_OPERATORS = {
    ast.Add: op.add,
    ast.Sub: op.sub,
    ast.Mult: op.mul,
    ast.Div: op.truediv,
    ast.FloorDiv: op.floordiv,
    ast.Mod: op.mod,
    ast.Pow: op.pow,
    ast.UAdd: op.pos,
    ast.USub: op.neg,
}


def normalize_text(text) -> str:
    if text is None:
        return ""
    text = str(text).lower().strip()
    text = text.replace("¿", "").replace("?", "")
    text = text.replace("¡", "").replace("!", "")
    text = text.replace("á", "a").replace("é", "e").replace("í", "i").replace("ó", "o").replace("ú", "u").replace("ñ", "n")
    text = re.sub(r"\s+", " ", text)
    return text


def tokenize(text: str) -> List[str]:
    text = normalize_text(text)
    tokens = re.findall(r"[a-z0-9]+", text)
    return [t for t in tokens if t and t not in STOPWORDS]



# =========================
# Inteligencia conversacional
# =========================
DOMAIN_SYNONYMS = {
    "horario": {"horario", "horarios", "hora", "abierto", "abren", "atienden", "atencion", "atención"},
    "precio": {"precio", "precios", "costo", "costos", "coste", "costes", "valor", "tarifa", "tarifas"},
    "ayuda": {"ayuda", "ayudar", "soporte", "asistencia", "ayudame", "ayúdame"},
    "detalle": {"detalle", "detalles", "amplia", "amplio", "ampliar", "explica", "explicame", "explícame", "describe"},
    "documento": {"documento", "documentos", "archivo", "archivos", "pdf", "txt"},
    "factura": {"factura", "facturación", "facturacion", "facturar"},
    "tramite": {"tramite", "trámite", "tramites", "trámites", "gestion", "gestión", "proceso"},
    "cliente": {"cliente", "clientes", "usuario", "usuarios", "persona", "personas"},
    "sancion": {"sancion", "sanciones", "multa", "multas", "penalidad", "penalidades"},
    "estado": {"estado", "estatus", "status", "situacion", "situación"},
    "ejemplo": {"ejemplo", "ejemplos", "muestra", "muestras", "modelo"},
    "definicion": {"definicion", "definición"},
}
SYNONYM_LOOKUP = {
    normalize_text(variant): canonical
    for canonical, variants in DOMAIN_SYNONYMS.items()
    for variant in variants
}

FOLLOWUP_WORDS = {
    "eso", "esto", "aquello", "anterior", "previo", "misma", "mismo", "mismos", "mismas",
    "detalle", "detalles", "amplia", "ampliar", "continua", "continuar", "seguir",
    "explica", "explicame", "explícame", "desarrolla", "profundiza", "más", "mas",
    "info", "informacion", "información", "acerca", "sobre", "lo anterior", "lo mismo",
}


SMALL_TALK_WORDS = {
    "hola", "buenas", "hello", "hey", "saludos", "gracias", "muchas gracias", "thanks",
    "adios", "adiós", "hasta luego", "chao", "bye",
}

GENERIC_REQUEST_WORDS = {
    "ayuda", "ayudame", "ayúdame", "explica", "explicame", "explícame", "dame", "dime",
    "busca", "encuentra", "muestra", "consulta", "describe", "resume", "resumeme", "resúmeme",
    "analiza", "compara", "interpreta", "ordena", "clasifica", "sugiere",
}

REQUEST_VERBS = {
    "quiero",
    "necesito",
    "busco",
    "dame",
    "muéstrame",
    "muestrame",
    "explica",
    "explícame",
    "explicame",
    "indica",
    "describe",
    "define",
    "comenta",
    "habla",
    "cuéntame",
    "cuentame",
    "informa",
    "ayuda"
}


QUESTION_WORDS = {
    "que", "qué", "quien", "quién", "cual", "cuál", "cuando", "cuándo",
    "donde", "dónde", "como", "cómo", "por", "porque", "porqué", "por que",
    "cuanto", "cuánto", "cuales", "cuáles"
}

HELP_TRIGGERS = {
    "ayuda", "ayudame", "ayúdame", "help", "soporte", "asistencia",
    "como uso", "cómo uso", "como funciona", "cómo funciona", "que puedes hacer", "qué puedes hacer"
}

ADMIN_TRIGGERS = {
    "admin", "administrador", "modo administrador", "contraseña", "configuracion", "configuración",
    "bases activas", "cargar bases", "guardar bases"
}

FOLLOWUP_TRIGGERS = {
    "eso", "esto", "aquello", "anterior", "previo", "lo anterior", "lo mismo",
    "detalle", "detalles", "amplia", "ampliar", "explica mas", "explica más",
    "mas detalle", "más detalle", "mas detalles", "más detalles", "continua", "continuar", "seguir"
}

DIRECT_QUESTION_PATTERNS = (
    r"^(?:que|qué|como|cómo|cual|cuál|cuando|cuándo|donde|dónde|quien|quién|porque|por que|cuanto|cuánto)\b",
    r"^(?:para que|para qué)\b",
    r"^(?:de que|de qué)\b",
    r"^(?:a que|a qué)\b",
)

def is_direct_question(query: str) -> bool:
    q = normalize_text(query)
    return any(re.search(pattern, q) for pattern in DIRECT_QUESTION_PATTERNS)

def expand_query_tokens(text: str) -> set:
    original_tokens = tokenize(text)
    expanded = set(original_tokens)
    for token in original_tokens:
        canonical = SYNONYM_LOOKUP.get(token)
        if canonical:
            expanded.add(canonical)
            # Añadimos también los sinónimos del grupo para ganar cobertura semántica.
            expanded.update(normalize_text(v) for v in DOMAIN_SYNONYMS.get(canonical, set()))
    return expanded


def contains_any(text: str, phrases) -> bool:
    q = normalize_text(text)
    for p in phrases:
        p_norm = normalize_text(p)
        if not p_norm:
            continue
        if " " in p_norm:
            if p_norm in q:
                return True
        else:
            if re.search(rf"\b{re.escape(p_norm)}\b", q):
                return True
    return False


def extract_content_tokens(text: str) -> List[str]:
    tokens = tokenize(text)
    return [t for t in tokens if t not in REQUEST_VERBS and t not in FOLLOWUP_WORDS]


def derive_topic_label(query: str, answer: str = "", last_context: Optional[dict] = None) -> str:
    query_tokens = extract_content_tokens(query)
    if query_tokens:
        return " ".join(query_tokens[:4]).strip()

    if last_context and last_context.get("topic"):
        return str(last_context.get("topic", "")).strip()

    answer_tokens = extract_content_tokens(answer)
    if answer_tokens:
        return " ".join(answer_tokens[:4]).strip()

    return normalize_text(query)[:50].strip()


def build_contextual_query(query: str, last_context: Optional[dict] = None, max_answer_tokens: int = 12) -> str:
    parts = [query]
    if last_context:
        topic = str(last_context.get("topic", "")).strip()
        if topic:
            parts.append(topic)

        answer = str(last_context.get("answer", "")).strip()
        if answer:
            answer_tokens = extract_content_tokens(answer)[:max_answer_tokens]
            if answer_tokens:
                parts.append(" ".join(answer_tokens))

    merged = " ".join(part for part in parts if part).strip()
    return re.sub(r"\s+", " ", merged)


def build_search_hint(query: str, bases: List[KnowledgeBase], limit: int = 4) -> List[str]:
    q_tokens = expand_query_tokens(query)
    suggestions = []
    if not q_tokens:
        return suggestions

    q_len = len(tokenize(query)) or 1
    for base in bases:
        for question in base.questions:
            q_base = expand_query_tokens(question)
            overlap = len(q_tokens & q_base)
            if overlap >= max(1, min(2, q_len)):
                suggestions.append(question)

    deduped = []
    seen = set()
    for item in suggestions:
        key = normalize_text(item)
        if key not in seen:
            seen.add(key)
            deduped.append(item)
    return deduped[:limit]


def _small_talk_response(query: str) -> Optional[str]:
    q = normalize_text(query)
    if not q:
        return None
    if contains_any(q, {"hola", "buenas", "hello", "hey", "saludos"}):
        return "Hola. Estoy listo para ayudarte a buscar respuestas, resolver cálculos o continuar una conversación sobre el tema anterior."
    if contains_any(q, {"gracias", "muchas gracias", "te agradezco", "thanks"}):
        return "Con gusto. Si necesitas algo más, lo revisamos."
    if contains_any(q, {"adios", "adiós", "hasta luego", "chao", "bye"}):
        return "Hasta luego. Cuando quieras, seguimos."
    return None


def classify_intent(query: str, bases: List[KnowledgeBase], last_context: Optional[dict] = None) -> dict:
    q = normalize_text(query)
    tokens = tokenize(q)
    content_tokens = extract_content_tokens(q)
    q_has_question_mark = "?" in query

    if not q:
        return {
            "route": "empty",
            "response": "Escribe una pregunta, una instrucción o una operación matemática.",
            "search": False,
        }

    small_talk = _small_talk_response(q)
    if small_talk:
        route = "greeting"
        if contains_any(q, {"gracias", "muchas gracias", "te agradezco", "thanks"}):
            route = "thanks"
        elif contains_any(q, {"adios", "adiós", "hasta luego", "chao", "bye"}):
            route = "goodbye"
        return {"route": route, "response": small_talk, "search": False}

    if contains_any(q, HELP_TRIGGERS) and len(tokens) <= 2:
        return {
            "route": "help",
            "response": (
                "Puedo ayudarte a buscar respuestas en las bases cargadas, resolver operaciones matemáticas y continuar el contexto de una pregunta anterior. "
                "También puedo darte sugerencias cuando no encuentre una coincidencia sólida."
            ),
            "search": False,
        }

    if contains_any(q, ADMIN_TRIGGERS) and len(tokens) <= 2:
        return {
            "route": "admin",
            "response": (
                "El modo administrador sirve para cargar, guardar y administrar las bases de conocimiento, además de revisar métricas y registros."
            ),
            "search": False,
        }

    if last_context and last_context.get("answer"):
        followup_signal = (
            contains_any(q, FOLLOWUP_TRIGGERS)
            or any(tok in FOLLOWUP_WORDS for tok in tokens)
            or (len(content_tokens) <= 3 and any(tok in GENERIC_REQUEST_WORDS for tok in tokens))
        )
        if followup_signal:
            expanded_query = build_contextual_query(query, last_context)
            return {
                "route": "followup",
                "response": "Voy a tomar como referencia el tema anterior para buscar una respuesta más precisa.",
                "search": True,
                "expanded_query": expanded_query,
            }

    question_like = q_has_question_mark or is_direct_question(query)
    request_like = any(t in REQUEST_VERBS or t in GENERIC_REQUEST_WORDS for t in tokens)
    has_meaningful_content = len(content_tokens) >= 1

    if question_like or request_like or has_meaningful_content:
        return {
            "route": "knowledge",
            "response": "",
            "search": True,
        }

    hints = build_search_hint(query, bases)
    if hints:
        hint_text = "\n".join(f"• {h}" for h in hints)
        return {
            "route": "clarify",
            "response": (
                "Aún no tengo claro qué necesitas. Estas opciones se parecen a lo que escribiste:\n\n"
                f"{hint_text}\n\n"
                "Si me das una palabra clave más concreta, lo intento de nuevo."
            ),
            "search": False,
        }

    return {
        "route": "clarify",
        "response": (
            "No estoy seguro de haber entendido la intención. Escríbelo con un poco más de detalle o usa una palabra clave del tema."
        ),
        "search": False,
    }


def format_no_match_response(query: str, bases: List[KnowledgeBase], last_context: Optional[dict] = None) -> str:
    hints = build_search_hint(query, bases)
    if hints:
        hint_text = "\n".join(f"• {h}" for h in hints)
        return (
            "No encontré una coincidencia sólida, pero estas preguntas parecen cercanas:\n\n"
            f"{hint_text}\n\n"
            "Si me das una palabra clave más concreta, lo intento de nuevo."
        )

    if last_context and last_context.get("answer"):
        topic = str(last_context.get("topic", "el tema anterior")).strip() or "el tema anterior"
        return (
            f"No encontré una coincidencia directa sobre {topic}. "
            "Si te refieres a eso, dime una palabra clave más específica y lo sigo buscando."
        )

    return (
        "No encontré una respuesta clara en las bases cargadas.\n\n"
        "Prueba con una pregunta más concreta, una palabra clave más específica o un par `p:... ?r:...` que incluya ese tema."
    )


def compose_match_response(answer: str, score: float, match_type: str) -> str:
    if match_type == "exacta":
        return f"Según la base, esto responde exactamente a tu consulta:\n\n{answer}"
    return (
        f"Esto es lo más cercano que encontré en las bases:\n\n"
        f"{answer}\n\n"
        f"> Confianza de coincidencia: {score*100:.0f}%"
    )

# =========================
# Bases de conocimiento
# =========================
def parse_qa_text(content: str) -> Dict[str, str]:
    memory: Dict[str, str] = {}
    if not content:
        return memory

    for raw_line in content.splitlines():
        line = raw_line.strip()
        if not line or "?r:" not in line:
            continue
        q, r = line.split("?r:", 1)
        q = normalize_text(q.replace("p:", "", 1).strip())
        r = r.strip().rstrip(".")
        if q and r:
            memory[q] = r
    return memory


def parse_qa_file(path: Path) -> Dict[str, str]:
    if not path.exists():
        return {}
    try:
        content = path.read_text(encoding="utf-8", errors="ignore").strip()
    except Exception:
        return {}
    return parse_qa_text(content)


@dataclass
class KnowledgeBase:
    name: str
    path: Path
    memory: Dict[str, str]

    @property
    def questions(self) -> List[str]:
        return list(self.memory.keys())


def load_primary_knowledge_bases(folder: Path, limit: int = MAX_PRIMARY_BASES) -> List[KnowledgeBase]:
    bases: List[KnowledgeBase] = []
    if folder.exists():
        for file_path in sorted(folder.glob("*.txt"))[:limit]:
            memory = parse_qa_file(file_path)
            if memory:
                bases.append(KnowledgeBase(file_path.stem, file_path, memory))
    return bases


def load_user_learning() -> Dict[str, str]:
    if LEARNING_FILE.exists():
        try:
            data = json.loads(LEARNING_FILE.read_text(encoding="utf-8"))
            if isinstance(data, dict):
                return {normalize_text(k): str(v).strip() for k, v in data.items() if str(k).strip()}
        except Exception:
            return {}
    return {}


def save_user_learning(data: Dict[str, str]) -> None:
    cleaned = {
        normalize_text(k): str(v).strip()
        for k, v in data.items()
        if str(k).strip() and str(v).strip()
    }
    ordered = dict(sorted(cleaned.items(), key=lambda item: item[0]))
    LEARNING_FILE.write_text(json.dumps(ordered, ensure_ascii=False, indent=2), encoding="utf-8")


def knowledge_base_from_mapping(name: str, mapping: Dict[str, str], path: Optional[Path] = None) -> KnowledgeBase:
    return KnowledgeBase(name=name, path=path or Path(name), memory=mapping)


# =========================
# Matemáticas seguras
# =========================
def looks_like_math(query) -> bool:
    if not query:
        return False
    q = normalize_text(query)
    return bool(re.search(r"\d", q)) and bool(re.search(r"[+\-*/()%]|\bmas\b|\bmenos\b|\bpor\b|\bentre\b", q))


def normalize_math_expression(text: str) -> str:
    expr = normalize_text(text)
    for pattern, replacement in MATH_WORDS:
        expr = re.sub(pattern, replacement, expr)

    prefixes = ["cuanto es", "cuánto es", "resuelve", "calcula", "resolver", "dime", "haz", "evalua", "evalúa"]
    for p in prefixes:
        expr = re.sub(rf"^\s*{re.escape(normalize_text(p))}\s*", "", expr)

    expr = expr.replace(" ", "")
    return expr


def safe_eval_math(expr: str) -> float:
    node = ast.parse(expr, mode="eval")

    def _eval(n):
        if isinstance(n, ast.Expression):
            return _eval(n.body)
        if isinstance(n, ast.Constant) and isinstance(n.value, (int, float)):
            return n.value
        if isinstance(n, ast.BinOp) and type(n.op) in _ALLOWED_OPERATORS:
            return _ALLOWED_OPERATORS[type(n.op)](_eval(n.left), _eval(n.right))
        if isinstance(n, ast.UnaryOp) and type(n.op) in _ALLOWED_OPERATORS:
            return _ALLOWED_OPERATORS[type(n.op)](_eval(n.operand))
        raise ValueError("Expresión no permitida")

    return _eval(node)


def format_math_result(value) -> str:
    try:
        if isinstance(value, float) and value.is_integer():
            return str(int(value))
    except Exception:
        pass
    return str(value)


# =========================
# Búsqueda híbrida/semántica simple
# =========================
def pair_search_text(question: str, answer: str) -> str:
    return f"{question} {answer}".strip()


def score_pair(query: str, question: str, answer: str) -> float:
    qn = normalize_text(query)
    qq = normalize_text(question)
    an = normalize_text(answer)

    q_tokens = tokenize(qn)
    q_expanded = expand_query_tokens(qn)
    qq_tokens = set(tokenize(qq))
    an_tokens = set(tokenize(an))
    pair_tokens = qq_tokens | an_tokens

    if not q_tokens:
        return 0.0

    pair_text = f"{qq} {an}".strip()
    q_set = set(q_tokens)
    q_expanded_set = set(q_expanded)

    # Coincidencia exacta de texto contra pregunta o respuesta
    if qn == qq or qn == an:
        return 1.0

    # Coincidencia por frase contenida en pregunta o respuesta
    if qn in qq:
        return 0.99

    if qn in an:
        return 0.90

    if qq in qn:
        return 0.92

    overlap = len(q_set & pair_tokens)
    expanded_overlap = len(q_expanded_set & pair_tokens)
    coverage = overlap / len(q_set)
    expanded_coverage = expanded_overlap / max(1, len(q_expanded_set))
    jaccard = overlap / max(1, len(q_set | pair_tokens))

    # Similitud difusa sobre pregunta, respuesta y conjunto completo
    seq_q = SequenceMatcher(None, qn, qq).ratio()
    seq_a = SequenceMatcher(None, qn, an).ratio()
    seq_pair = SequenceMatcher(None, qn, pair_text).ratio()

    # Premio fuerte si los tokens del usuario están dentro de la pregunta o respuesta
    question_tokens = set(qq_tokens)
    answer_tokens = set(an_tokens)

    if q_set.issubset(question_tokens):
        return 0.98

    if q_set.issubset(answer_tokens):
        return 0.92

    # Si la consulta es muy corta, exigimos presencia textual o token exacto
    if len(q_tokens) == 1:
        token = q_tokens[0]
        if token in qq_tokens or token in an_tokens:
            return 0.96
        return max(
            coverage * 0.82,
            expanded_coverage * 0.80,
            jaccard * 0.74,
            seq_q * 0.44,
            seq_a * 0.44,
            seq_pair * 0.50,
        )

    # Combinación conservadora: prioriza cobertura de tokens, expansión y similitud difusa
    score = (
        0.42 * coverage +
        0.18 * expanded_coverage +
        0.16 * jaccard +
        0.12 * seq_q +
        0.08 * seq_a +
        0.04 * seq_pair
    )

    # Bonus por coincidencias reales de palabras
    if overlap >= 2:
        score += 0.05

    # Bonus pequeño si además la expansión semántica ayuda
    if expanded_overlap > overlap:
        score += 0.03

    # Nunca dejar que SequenceMatcher domine por sí solo
    score = max(score, coverage * 0.90, expanded_coverage * 0.88, jaccard * 0.95)
    return min(score, 1.0)


def search_knowledge(query: str, bases: List[KnowledgeBase], threshold: float = DEFAULT_THRESHOLD) -> Tuple[Optional[str], str, float, str]:
    """
    Devuelve: respuesta, base, score, tipo
    tipo: exacta | aproximada | ninguna
    """
    q = normalize_text(query)
    q_tokens = set(tokenize(q))

    best_answer = None
    best_source = ""
    best_score = 0.0
    best_question = ""
    best_overlap = -1

    if not q_tokens:
        return None, "", 0.0, "ninguna"

    # 1) Coincidencia exacta contra pregunta o respuesta
    for base in bases:
        for question, answer in base.memory.items():
            nq = normalize_text(question)
            na = normalize_text(answer)
            if q == nq or q == na:
                return answer, base.name, 1.0, "exacta"

    # 2) Búsqueda por palabras clave con similitud conservadora
    for base in bases:
        for question, answer in base.memory.items():
            score = score_pair(q, question, answer)

            question_tokens = set(tokenize(question))
            answer_tokens = set(tokenize(answer))
            overlap = len(q_tokens & (question_tokens | answer_tokens))

            if score > best_score:
                best_score = score
                best_answer = answer
                best_source = base.name
                best_question = question
                best_overlap = overlap
                continue

            if abs(score - best_score) < 0.03:
                better = False
                if overlap > best_overlap:
                    better = True
                elif overlap == best_overlap:
                    if best_question and len(question) < len(best_question):
                        better = True
                    elif best_question and len(question) == len(best_question):
                        better = score > best_score

                if better:
                    best_score = score
                    best_answer = answer
                    best_source = base.name
                    best_question = question
                    best_overlap = overlap

    if best_answer and best_score >= threshold:
        return best_answer, best_source, best_score, "aproximada"

    return None, "", 0.0, "ninguna"


# =========================
# Registro y exportación
# =========================
def ensure_log_file() -> None:
    if not LOG_FILE.exists():
        with LOG_FILE.open("w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["timestamp", "query", "type", "source", "score", "answer"])


def log_query(query: str, qtype: str, source: str, score: float, answer: str) -> None:
    ensure_log_file()
    with LOG_FILE.open("a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            datetime.now().isoformat(timespec="seconds"),
            query,
            qtype,
            source,
            f"{score:.3f}",
            answer,
        ])


def export_conversation_text(messages: List[Dict[str, str]]) -> str:
    lines = []
    for msg in messages:
        role = "Usuario" if msg["role"] == "user" else "Bot"
        lines.append(f"{role}: {msg['content']}")
    return "\n".join(lines)


def read_log_stats() -> Tuple[Counter, Counter]:
    """
    Devuelve:
      - contador por tipo de consulta
      - contador por pregunta normalizada
    """
    type_counter = Counter()
    question_counter = Counter()

    if not LOG_FILE.exists():
        return type_counter, question_counter

    try:
        with LOG_FILE.open("r", newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                qtype = row.get("type", "")
                q = normalize_text(row.get("query", ""))
                if qtype:
                    type_counter[qtype] += 1
                if q:
                    question_counter[q] += 1
    except Exception:
        pass

    return type_counter, question_counter


# =========================
# Carga de bases por archivos
# =========================
def imported_bases_from_uploads(uploaded_files) -> List[KnowledgeBase]:
    bases = []
    if not uploaded_files:
        return bases

    for up in uploaded_files:
        try:
            raw = up.getvalue().decode("utf-8", errors="ignore")
        except Exception:
            raw = ""
        memory = parse_qa_text(raw)
        if memory:
            bases.append(KnowledgeBase(name=Path(up.name).stem, path=Path(up.name), memory=memory))
    return bases


def save_uploaded_bases(uploaded_files) -> List[str]:
    KNOWLEDGE_FOLDER.mkdir(exist_ok=True)
    saved = []
    for up in uploaded_files or []:
        target = KNOWLEDGE_FOLDER / Path(up.name).name
        try:
            target.write_bytes(up.getvalue())
            saved.append(target.name)
        except Exception:
            pass
    return saved


def dedupe_bases(bases: List[KnowledgeBase]) -> List[KnowledgeBase]:
    seen = set()
    result = []
    for base in bases:
        key = base.name.lower()
        if key not in seen:
            result.append(base)
            seen.add(key)
    return result


# =========================
# App
# =========================
st.set_page_config(page_title=APP_TITLE, page_icon="🤖", layout="wide")
st.title("🤖 Chatbot híbrido PRO")
st.caption("Búsqueda exacta, aproximada y por palabras clave. Con métricas, logs, aprendizaje, exportación y carga dinámica de bases.")

if "messages" not in st.session_state:
    st.session_state.messages = []

if "stats" not in st.session_state:
    st.session_state.stats = {
        "total": 0,
        "exacta": 0,
        "aproximada": 0,
        "math": 0,
        "no_match": 0,
    }

if "admin_mode" not in st.session_state:
    st.session_state.admin_mode = False

if "uploaded_bases" not in st.session_state:
    st.session_state.uploaded_bases = []

if "import_message" not in st.session_state:
    st.session_state.import_message = ""

if "conversation_context" not in st.session_state:
    st.session_state.conversation_context = {
        "topic": "",
        "question": "",
        "answer": "",
        "source": "",
        "type": "",
    }

# Cargar fuentes
primary_bases = load_primary_knowledge_bases(KNOWLEDGE_FOLDER, limit=MAX_PRIMARY_BASES)
learned_memory = load_user_learning()
learned_base = knowledge_base_from_mapping("aprendizaje_usuario", learned_memory, LEARNING_FILE) if learned_memory else None
upload_bases = imported_bases_from_uploads(st.session_state.uploaded_bases)

all_bases = primary_bases[:]
if learned_base:
    all_bases.append(learned_base)
all_bases.extend(upload_bases)
all_bases = dedupe_bases(all_bases)

# Sidebar superior: administración, carga dinámica y aprendizaje
with st.sidebar:
    st.header("Configuración")
    threshold = st.slider("Umbral de coincidencia", 0.40, 0.95, float(DEFAULT_THRESHOLD), 0.01)

    st.markdown("---")
    st.subheader("Carga de bases")
    uploaded = st.file_uploader(
        "Sube archivos .txt con formato p:... ?r:...",
        type=["txt"],
        accept_multiple_files=True,
        key="uploader_bases",
    )

    col_a, col_b = st.columns(2)
    with col_a:
        if st.button("Cargar temporalmente"):
            st.session_state.uploaded_bases = uploaded or []
            st.session_state.import_message = "Bases cargadas temporalmente."
    with col_b:
        if st.button("Guardar en carpeta bases"):
            if uploaded:
                saved = save_uploaded_bases(uploaded)
                st.session_state.import_message = f"Guardadas: {', '.join(saved)}" if saved else "No se pudo guardar ningún archivo."
                st.session_state.uploaded_bases = uploaded
            else:
                st.session_state.import_message = "No hay archivos para guardar."

    if st.session_state.import_message:
        st.info(st.session_state.import_message)

    st.markdown("---")
    st.subheader("Modo administrador")
    if not st.session_state.admin_mode:
        pwd = st.text_input("Contraseña", type="password", key="admin_pwd")
        if st.button("Entrar"):
            if pwd == ADMIN_PASSWORD:
                st.session_state.admin_mode = True
                st.session_state.pop("admin_pwd", None)
                st.rerun()
            else:
                st.error("Contraseña incorrecta.")
    else:
        st.success("Administrador activo")
        learned_snapshot = load_user_learning()
        st.caption(f"Pares aprendidos activos: {len(learned_snapshot)}")

        st.markdown("---")
        st.subheader("Aprendizaje")

        learn_question = st.text_input(
            "Pregunta nueva",
            key="learn_question"
        )

        learn_answer = st.text_area(
            "Respuesta",
            key="learn_answer"
        )

        col_l1, col_l2 = st.columns(2)
        with col_l1:
            if st.button("Guardar aprendizaje"):
                question = learn_question.strip()
                answer = learn_answer.strip()
                if question and answer:
                    learned = load_user_learning()
                    learned[question] = answer
                    save_user_learning(learned)
                    st.session_state.pop("learn_question", None)
                    st.session_state.pop("learn_answer", None)
                    st.success("Aprendizaje guardado correctamente.")
                    st.rerun()
                else:
                    st.warning("Debes completar pregunta y respuesta.")
        with col_l2:
            if st.button("Limpiar campos"):
                st.session_state.pop("learn_question", None)
                st.session_state.pop("learn_answer", None)
                st.rerun()

        if learned_snapshot:
            with st.expander("Ver aprendizajes guardados"):
                for q_saved, a_saved in list(learned_snapshot.items())[:12]:
                    st.write(f"• **{q_saved}** → {a_saved}")

            delete_options = [""] + sorted(learned_snapshot.keys())
            delete_key = st.selectbox(
                "Eliminar un aprendizaje",
                options=delete_options,
                key="delete_learning_select"
            )
            if st.button("Eliminar seleccionado"):
                if delete_key and delete_key in learned_snapshot:
                    learned_snapshot.pop(delete_key, None)
                    save_user_learning(learned_snapshot)
                    st.session_state.pop("delete_learning_select", None)
                    st.success("Aprendizaje eliminado.")
                    st.rerun()
                else:
                    st.warning("Selecciona un aprendizaje para eliminar.")

        if st.button("Salir de admin"):
            st.session_state.admin_mode = False
            st.session_state.pop("admin_pwd", None)
            st.session_state.pop("learn_question", None)
            st.session_state.pop("learn_answer", None)
            st.session_state.pop("delete_learning_select", None)
            st.rerun()

    st.markdown("---")
    st.subheader("Bases activas")
    if all_bases:
        for base in all_bases:
            st.write(f"• **{base.name}** — {len(base.memory)} pares")
    else:
        st.warning("No hay bases cargadas.")

    st.markdown("---")
    st.subheader("Ejemplo de matemáticas")
    st.code("2 + 3 * 5\ncuanto es 12 entre 3\n(8 - 2) / 2", language="text")

# Procesamiento principal del chat
if not all_bases:
    st.error("No se cargó ninguna base de conocimiento.")
else:
    # Historial
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    user_query = st.chat_input("Escribe tu pregunta, una instrucción o una operación matemática...")

    if user_query:
        st.session_state.stats["total"] += 1

        user_query_clean = user_query.strip()
        st.session_state.messages.append({"role": "user", "content": user_query_clean})

        with st.chat_message("user"):
            st.markdown(user_query_clean)

        response_text = ""
        meta = ""
        qtype = "no_match"
        source_name = ""
        score = 0.0

        intent = classify_intent(user_query_clean, all_bases, st.session_state.conversation_context)

        if looks_like_math(user_query_clean):
            try:
                expr = normalize_math_expression(user_query_clean)
                result = safe_eval_math(expr)
                response_text = f"El resultado es **{format_math_result(result)}**."
                meta = f"Operación detectada: `{expr}`"
                st.session_state.stats["math"] += 1
                qtype = "math"
                source_name = "calculadora"
                score = 1.0
                st.session_state.conversation_context = {
                    "topic": derive_topic_label(user_query_clean, response_text, st.session_state.conversation_context),
                    "question": user_query_clean,
                    "answer": response_text,
                    "source": source_name,
                    "type": qtype,
                }
            except Exception as e:
                response_text = "Detecté una operación, pero no pude evaluarla de forma segura."
                meta = f"Error: {e}"
                st.session_state.stats["no_match"] += 1
                qtype = "no_match"
                source_name = "calculadora"
                score = 0.0
        elif intent["route"] in {"greeting", "thanks", "goodbye", "help", "admin", "clarify", "empty"} and not intent.get("search"):
            response_text = intent["response"]
            meta = "Respuesta conversacional."
            qtype = intent["route"]
            source_name = "conversacion"
            score = 1.0 if intent["route"] in {"greeting", "thanks", "goodbye"} else 0.5
        else:
            expanded_query = intent.get("expanded_query", user_query_clean)
            answer, source_name, score, match_type = search_knowledge(expanded_query, all_bases, threshold=threshold)
            if answer:
                response_text = compose_match_response(answer, score, match_type)
                if match_type == "exacta":
                    st.session_state.stats["exacta"] += 1
                    meta = f"Fuente: **{source_name}** | coincidencia exacta"
                    qtype = "exacta"
                else:
                    st.session_state.stats["aproximada"] += 1
                    meta = f"Fuente: **{source_name}** | coincidencia aproximada"
                    qtype = "aproximada"

                st.session_state.conversation_context = {
                    "topic": derive_topic_label(user_query_clean, answer, st.session_state.conversation_context),
                    "question": user_query_clean,
                    "answer": answer,
                    "source": source_name,
                    "type": qtype,
                }
            else:
                st.session_state.stats["no_match"] += 1
                response_text = format_no_match_response(user_query_clean, all_bases, st.session_state.conversation_context)
                meta = "Sin coincidencia."
                qtype = "no_match"

        # Registrar consulta en CSV
        log_query(user_query_clean, qtype, source_name, score, response_text)

        # Guardar respuesta
        st.session_state.messages.append({"role": "assistant", "content": response_text})

        with st.chat_message("assistant"):
            st.markdown(response_text)
            if meta:
                st.caption(meta)

        # sugerencias rápidas
        if qtype == "no_match" and not looks_like_math(user_query_clean):
            suggestions = build_search_hint(user_query_clean, all_bases, limit=4)
            if suggestions:
                st.info("Quizás quisiste decir:")
                for s in suggestions:
                    st.write(f"• {s}")

# Panel de analítica y exportación al final para que refleje la consulta actual
type_counter, question_counter = read_log_stats()

with st.sidebar:
    st.markdown("---")
    st.subheader("📊 Estadísticas")
    stats = st.session_state.stats
    st.metric("Consultas", stats["total"])
    st.metric("Exactas", stats["exacta"])
    st.metric("Aproximadas", stats["aproximada"])
    st.metric("Matemáticas", stats["math"])
    st.metric("Sin respuesta", stats["no_match"])

    st.markdown("---")
    st.subheader("📈 Gráfico de desempeño")
    chart_data = {
        "Exactas": stats["exacta"],
        "Aproximadas": stats["aproximada"],
        "Matemáticas": stats["math"],
        "Sin respuesta": stats["no_match"],
    }
    st.bar_chart(chart_data)

    st.markdown("---")
    st.subheader("🔝 Preguntas más frecuentes")
    if question_counter:
        for q, count in question_counter.most_common(5):
            st.write(f"• {q} — {count}")
    else:
        st.caption("Aún no hay consultas registradas.")

    st.markdown("---")
    st.subheader("📥 Exportar")
    conversation_text = export_conversation_text(st.session_state.messages)
    st.download_button(
        "Descargar conversación",
        data=conversation_text,
        file_name=CONVERSATION_EXPORT_FILE,
        mime="text/plain",
    )

    if LOG_FILE.exists():
        with open(LOG_FILE, "rb") as f:
            st.download_button(
                "Descargar consultas CSV",
                data=f,
                file_name=LOG_FILE.name,
                mime="text/csv",
            )

    if LEARNING_FILE.exists():
        with open(LEARNING_FILE, "rb") as f:
            st.download_button(
                "Descargar aprendizaje JSON",
                data=f,
                file_name=LEARNING_FILE.name,
                mime="application/json",
            )

    st.markdown("---")
    st.subheader("📚 Resumen de tipos")
    if type_counter:
        st.write(f"Exactas: {type_counter.get('exacta', 0)}")
        st.write(f"Aproximadas: {type_counter.get('aproximada', 0)}")
        st.write(f"Matemáticas: {type_counter.get('math', 0)}")
        st.write(f"Sin respuesta: {type_counter.get('no_match', 0)}")
