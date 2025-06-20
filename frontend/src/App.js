import React from 'react';
import {
  AppBar, Toolbar, Typography, IconButton, Container, Grid, Card, CardContent,
  TextField, Button, Box, Divider, Table, TableBody, TableCell, TableHead, TableRow, Paper
} from '@mui/material';
import MedicalServicesIcon from '@mui/icons-material/MedicalServices';
import InfoOutlinedIcon from '@mui/icons-material/InfoOutlined';

function App() {
  return (
    <Box sx={{ minHeight: '100vh', bgcolor: '#f4f8fb' }}>
      <AppBar position="static" color="primary" elevation={2}>
        <Toolbar>
          <IconButton edge="start" color="inherit" aria-label="logo" sx={{ mr: 2 }}>
            <MedicalServicesIcon fontSize="large" />
          </IconButton>
          <Typography variant="h5" sx={{ flexGrow: 1, fontWeight: 700, letterSpacing: 1 }}>
            AI Doctor
          </Typography>
        </Toolbar>
      </AppBar>
      <Container maxWidth="lg" sx={{ mt: 6, mb: 4 }}>
        <Grid container spacing={5} alignItems="flex-start">
          {/* Patient Input */}
          <Grid item xs={12} md={6}>
            <Card elevation={4} sx={{ borderRadius: 3 }}>
              <CardContent>
                <Typography variant="h6" gutterBottom sx={{ fontWeight: 600 }}>
                  Patient Input
                </Typography>
                <TextField
                  label="Symptoms"
                  multiline
                  rows={4}
                  fullWidth
                  margin="normal"
                  variant="outlined"
                  sx={{ bgcolor: 'white', borderRadius: 1 }}
                />
                <TextField
                  label="Medical History (optional)"
                  multiline
                  rows={2}
                  fullWidth
                  margin="normal"
                  variant="outlined"
                  sx={{ bgcolor: 'white', borderRadius: 1 }}
                />
                <Button
                  variant="outlined"
                  component="label"
                  sx={{ mt: 2, borderRadius: 2, fontWeight: 500 }}
                  fullWidth
                  color="primary"
                >
                  Upload CT scans or reports
                  <input type="file" hidden multiple />
                </Button>
                <Button
                  variant="contained"
                  color="primary"
                  sx={{ mt: 3, borderRadius: 2, fontWeight: 600, boxShadow: 2, py: 1.2 }}
                  fullWidth
                  size="large"
                >
                  Analyze
                </Button>
              </CardContent>
            </Card>
          </Grid>
          {/* Results */}
          <Grid item xs={12} md={6}>
            <Card elevation={4} sx={{ borderRadius: 3 }}>
              <CardContent>
                <Typography variant="h6" gutterBottom sx={{ fontWeight: 600 }}>
                  AI Diagnosis & Recommendations
                </Typography>
                <Box sx={{ mt: 2, minHeight: 120 }}>
                  <Typography variant="body1" color="text.secondary">
                    {/* Placeholder for diagnosis/results */}
                    Diagnosis and recommendations will appear here after analysis.
                  </Typography>
                </Box>
                <Divider sx={{ my: 2 }} />
                <Typography variant="h6" sx={{ fontWeight: 600, mb: 1 }}>
                  Medicines Table
                </Typography>
                <Paper variant="outlined" sx={{ bgcolor: '#f9fafb' }}>
                  <Table size="small">
                    <TableHead>
                      <TableRow>
                        <TableCell>Medicine</TableCell>
                        <TableCell>Type</TableCell>
                        <TableCell>Dosage</TableCell>
                        <TableCell>Side Effects</TableCell>
                      </TableRow>
                    </TableHead>
                    <TableBody>
                      {/* Example row */}
                      <TableRow>
                        <TableCell>Paracetamol</TableCell>
                        <TableCell>Indian</TableCell>
                        <TableCell>500mg</TableCell>
                        <TableCell>Rare: rash, nausea</TableCell>
                      </TableRow>
                    </TableBody>
                  </Table>
                </Paper>
                <Box sx={{ display: 'flex', alignItems: 'center', mt: 2 }}>
                  <InfoOutlinedIcon color="info" sx={{ mr: 1 }} />
                  <Typography variant="body2" color="text.secondary">
                    This is not a substitute for professional medical advice.
                  </Typography>
                </Box>
              </CardContent>
            </Card>
          </Grid>
        </Grid>
      </Container>
      <Box component="footer" sx={{ py: 2, bgcolor: '#e3eaf2', textAlign: 'center', borderTop: 1, borderColor: '#d1d9e6' }}>
        <Typography variant="body2" color="text.secondary">
          &copy; {new Date().getFullYear()} AI Doctor. All rights reserved.
        </Typography>
      </Box>
    </Box>
  );
}

export default App;
