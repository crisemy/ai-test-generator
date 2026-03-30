# Guía Rápida - JIRA Seed

## 📋 Qué se creó

```
scripts/
├── seed_jira.py                    # Script principal (sin LLM)
└── README.md                       # Documentación

assets/
└── jira-seed-template.json        # Plantilla con 5 tickets de ejemplo

.github/workflows/
└── seed-jira.yml                   # GitHub Actions (opcional)
```

## 🚀 Uso Rápido

### 1. Asegurate que `.env` tiene tus credenciales JIRA:

```bash
JIRA_BASE_URL=https://tu-empresa.atlassian.net
JIRA_EMAIL=tu-email@empresa.com
JIRA_API_TOKEN=<tu-api-token>
```

### 2. Prueba sin crear nada (DRY RUN):

```bash
python scripts/seed_jira.py --dry-run
```

Verás algo como:
```
🔗 Testing JIRA connection...
✅ JIRA connection successful

🔄 DRY-RUN   | SIGNUP-001   | Would create User Signup with Email and Password
🔄 DRY-RUN   | SIGNUP-002   | Would create User Signup with OAuth (Google/GitHub)
...

💡 This was a DRY RUN. Run without --dry-run to create issues.
```

### 3. Si todo se ve bien, crea los tickets:

```bash
python scripts/seed_jira.py
```

Output:
```
✅ SUCCESS   | SAAS-123     | Created SAAS-123
           → https://tu-empresa.atlassian.net/browse/SAAS-123
✅ SUCCESS   | SAAS-124     | Created SAAS-124
✅ SUCCESS   | SAAS-125     | Created SAAS-125
✅ SUCCESS   | SAAS-126     | Created SAAS-126
✅ SUCCESS   | SAAS-127     | Created SAAS-127

============================================================
Summary: 5 created, 0 errors, 0 skipped, 0 dry-run
============================================================
```

## 🎯 Personalizar Plantilla

Edita `assets/jira-seed-template.json`:

```json
{
  "project_key": "TUPROYECTO",        // Cambiar clave de proyecto
  "issue_type": "Story",               // Story, Task, Bug, etc.
  "labels": ["ai-seed", "tu-etiqueta"],
  "tickets": [
    {
      "key": "TU-001",
      "summary": "Tu resumen",
      "description": "Tu descripción",
      "priority": "High",              // Low, Medium, High
      "reporter": "Tu nombre"
    }
  ]
}
```

Luego ejecuta:
```bash
python scripts/seed_jira.py --template assets/tu-plantilla.json
```

## 🔍 Filtrar en JIRA

En JIRA JQL, todos los tickets seeded tienen etiqueta `ai-seed`:

```
labels = ai-seed
```

Perfecto para filtrar, actualizar o eliminar lotes.

## ⚙️ GitHub Actions (Opcional)

Para automatizar:

1. **Agregar secretos** en GitHub (Settings → Secrets):
   - `JIRA_BASE_URL`
   - `JIRA_EMAIL`  
   - `JIRA_API_TOKEN`

2. **Ejecutar manualmente**:
   - Ve a Actions → Seed JIRA Issues → Run workflow

3. **Programado** (cada primer día del mes):
   - Ya está en `.github/workflows/seed-jira.yml`
   - Edita el `cron` si quieres otra frecuencia

## ✨ Ventajas

✅ **Sin LLM**: No gasta tokens de API de IA  
✅ **Rápido**: Segundos para crear 5-100 tickets  
✅ **Seguro**: Dry-run para verificar antes de crear  
✅ **Flexible**: Plantillas JSON reutilizables  
✅ **Automatizable**: Cron, manual, o CI/CD  
✅ **Filtrable**: Etiqueta `ai-seed` para identificar  

## 🐛 Problemas Comunes

| Error | Solución |
|-------|----------|
| `401 Unauthorized` | Verifica email y API token en `.env` |
| `404 Not Found` | URL de JIRA debe ser `https://domain.atlassian.net` |
| `project_key not found` | El proyecto no existe o no tienes acceso |
| `Invalid issue type` | Verifica que Story/Task exista en tu proyecto |

---

**Next steps**: 
- [ ] Configurar `.env` con credenciales JIRA
- [ ] Ejecutar `python scripts/seed_jira.py --dry-run`
- [ ] Si OK, ejecutar `python scripts/seed_jira.py`
- [ ] Verificar tickets en JIRA (filtro: `labels = ai-seed`)
- [ ] (Opcional) Configurar secretos en GitHub para automatización
