use serde::{Deserialize, Serialize};

/// Stores users organizations
#[derive(Serialize, Deserialize, Debug)]
pub struct Organization {
    /// The organization id
    pub id: String,
    /// The users username
    pub name: String,
}

impl Organization {
    /// Export the organization as json
    pub fn as_json(&self) -> Result<String, Box<dyn std::error::Error>> {
        let serialized = match serde_json::to_string(&self) {
            Ok(s) => s,
            Err(e) => {
                return Err(Box::new(e));
            }
        };
        Ok(serialized)
    }
}

/// Stores Organization
#[derive(Serialize, Deserialize, Debug)]
pub struct Organizations {
    /// An Array of Organizations
    pub organizations: Vec<Organization>,
}

impl Organizations {
    /// Export the organizations as json
    pub fn as_json(&self) -> Result<String, Box<dyn std::error::Error>> {
        let serialized = match serde_json::to_string(&self) {
            Ok(s) => s,
            Err(e) => {
                return Err(Box::new(e));
            }
        };
        Ok(serialized)
    }
}

/// Stores the Proof Response
#[derive(Debug, Deserialize, Serialize)]
pub struct Proof {
    /// stores the artifact
    pub artifact: Option<Artifact>,
    /// stores the Proof Id
    pub id: String,
    /// stores the instances
    pub instances: Option<Vec<String>>,
    /// stores the proofs
    pub proof: Option<String>,
    /// stores the status
    pub status: Option<String>,
    /// stores the transcript type
    #[serde(rename = "transcriptType")]
    pub transcript_type: Option<String>,
}

impl Proof {
    /// Export the proof as json
    pub fn as_json(&self) -> Result<String, Box<dyn std::error::Error>> {
        let serialized = match serde_json::to_string(&self) {
            Ok(s) => s,
            Err(e) => {
                return Err(Box::new(e));
            }
        };
        Ok(serialized)
    }
}

/// Stores the Artifacts
#[derive(Debug, Deserialize, Serialize)]
pub struct Artifact {
    ///stores the aritfact id
    pub id: Option<String>,
    /// stores the name of the artifact
    pub name: Option<String>,
}

impl Artifact {
    /// Export the artifact as json
    pub fn as_json(&self) -> Result<String, Box<dyn std::error::Error>> {
        let serialized = match serde_json::to_string(&self) {
            Ok(s) => s,
            Err(e) => {
                return Err(Box::new(e));
            }
        };
        Ok(serialized)
    }
}

#[cfg(feature = "python-bindings")]
impl pyo3::ToPyObject for Artifact {
    fn to_object(&self, py: pyo3::Python) -> pyo3::PyObject {
        let dict = pyo3::types::PyDict::new(py);
        dict.set_item("id", &self.id).unwrap();
        dict.set_item("name", &self.name).unwrap();
        dict.into()
    }
}

#[cfg(feature = "python-bindings")]
impl pyo3::ToPyObject for Proof {
    fn to_object(&self, py: pyo3::Python) -> pyo3::PyObject {
        let dict = pyo3::types::PyDict::new(py);
        dict.set_item("artifact", &self.artifact).unwrap();
        dict.set_item("id", &self.id).unwrap();
        dict.set_item("instances", &self.instances).unwrap();
        dict.set_item("proof", &self.proof).unwrap();
        dict.set_item("status", &self.status).unwrap();
        dict.set_item("strategy", &self.strategy).unwrap();
        dict.set_item("transcript_type", &self.transcript_type)
            .unwrap();
        dict.into()
    }
}

#[cfg(feature = "python-bindings")]
impl pyo3::ToPyObject for Organizations {
    fn to_object(&self, py: pyo3::Python) -> pyo3::PyObject {
        let dict = pyo3::types::PyDict::new(py);
        dict.set_item("organizations", &self.organizations).unwrap();
        dict.into()
    }
}

#[cfg(feature = "python-bindings")]
impl pyo3::ToPyObject for Organization {
    fn to_object(&self, py: pyo3::Python) -> pyo3::PyObject {
        let dict = pyo3::types::PyDict::new(py);
        dict.set_item("id", &self.id).unwrap();
        dict.set_item("name", &self.name).unwrap();
        dict.into()
    }
}
