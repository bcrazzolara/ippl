#ifndef IPPL_STRUCTURE_FORMATION_MANAGER_H
#define IPPL_STRUCTURE_FORMATION_MANAGER_H

#include <memory>
#include <iostream>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>
#include <iomanip>

using namespace std;

#include "GravityFieldContainer.hpp"
#include "GravityFieldSolver.hpp"
#include "GravityLoadBalancer.hpp"
#include "GravityManager.h"
#include "Manager/BaseManager.h"
#include "GravityParticleContainer.hpp"
#include "Random/Distribution.h"
#include "Random/InverseTransformSampling.h"
#include "Random/NormalDistribution.h"
#include "Random/Randn.h"

string folder = "/data/user/crazzo_b/IPPL/ippl/build_openmp/alpine/data/lsf_16/";

using view_type = typename ippl::detail::ViewType<ippl::Vector<double, Dim>, 1>::view_type;
//typedef ippl::ParticleSpatialLayout<double, Dim> PLayout_t;

template <typename T, unsigned Dim>
class StructureFormationManager : public GravityManager<T, Dim> {
public:
    using ParticleContainer_t = ParticleContainer<T, Dim>;
    using FieldContainer_t = FieldContainer<T, Dim>;
    using FieldSolver_t= FieldSolver<T, Dim>;
    using LoadBalancer_t= LoadBalancer<T, Dim>;

    StructureFormationManager(size_type totalP_, int nt_, Vector_t<int, Dim> &nr_,
                       double lbt_, std::string& solver_, std::string& stepMethod_)
        : GravityManager<T, Dim>(totalP_, nt_, nr_, lbt_, solver_, stepMethod_){}

    ~StructureFormationManager(){}

    void pre_run() override {
        Inform mes("Pre Run");

        if (this->solver_m == "OPEN") {
            throw IpplException("StructureFormation", "Open boundaries solver incompatible with this simulation!");
        }

        // Grid 
        for (unsigned i = 0; i < Dim; i++) {
            this->domain_m[i] = ippl::Index(this->nr_m[i]);
        }

        this->decomp_m.fill(true);
        this->Hubble0 =  0.1; // h * km/sec/kpc  (h = 0.7, H = 0.07)
        this->G = 4.30071e04; // kpc km^2 /s^2 / M_Sun e10
        this->z_m = 63;
        this->InitialiseTime();

        this->rmin_m  = 0.0;
        this->rmax_m  = 50000.0; // kpc/h
        double Vol = std::reduce(this->rmax_m.begin(), this->rmax_m.end(), 1., std::multiplies<double>());
        this->M_m = this->rho_crit0 * Vol * this->O_m; // 1e10 M_Sun
        mes << "total mass: " << this->M_m << endl;
        mes << "mass of a single particle " << this->M_m/this->totalP_m << endl;

        this->hr_m = this->rmax_m / this->nr_m;
        this->origin_m = this->rmin_m;
        this->it_m     = 0;


        mes << "Discretization:" << endl
          << "nt " << this->nt_m << ", Np = " << this->totalP_m << ", grid = " << this->nr_m << endl;

        this->isAllPeriodic_m = true;

        this->setFieldContainer( std::make_shared<FieldContainer_t>( this->hr_m, this->rmin_m, this->rmax_m, this->decomp_m, this->domain_m, this->origin_m, this->isAllPeriodic_m) );

        this->setParticleContainer( std::make_shared<ParticleContainer_t>( this->fcontainer_m->getMesh(), this->fcontainer_m->getFL()) );

        this->fcontainer_m->initializeFields(this->solver_m);

        this->setFieldSolver( std::make_shared<FieldSolver_t>( this->solver_m, &this->fcontainer_m->getRho(), &this->fcontainer_m->getF(), &this->fcontainer_m->getPhi()) );

        this->fsolver_m->initSolver();

        this->setLoadBalancer( std::make_shared<LoadBalancer_t>( this->lbt_m, this->fcontainer_m, this->pcontainer_m, this->fsolver_m) );

        readParticles(); // defines particle positions, velocities

        static IpplTimings::TimerRef DummySolveTimer  = IpplTimings::getTimer("solveWarmup");
        IpplTimings::startTimer(DummySolveTimer);
        
        this->fcontainer_m->getRho() = 0.0;
        //this->fcontainer_m->getRho().write();
        //ippl::Comm->barrier();
        this->fsolver_m->runSolver();

        IpplTimings::stopTimer(DummySolveTimer);
        this->par2grid();

        static IpplTimings::TimerRef SolveTimer = IpplTimings::getTimer("solve");
        IpplTimings::startTimer(SolveTimer);

        this->fsolver_m->runSolver();

        IpplTimings::stopTimer(SolveTimer);

        this->grid2par();

        this->dump();
        
        mes << "Done";
    }
    
    void readParticles() {
        Inform mes("Reading Particles");

        size_type nloc = this->totalP_m / ippl::Comm->size(); 
        mes << "Local number of particles: " << nloc << endl; 
        std::shared_ptr<ParticleContainer_t> pc = this->pcontainer_m;
        pc->create(nloc);  
        pc->m = this->M_m/this->totalP_m;

        this->fcontainer_m->getRho() = 0.0;

        // Load Balancer Initialisation
        auto *mesh = &this->fcontainer_m->getMesh();
        auto *FL = &this->fcontainer_m->getFL();
        if ((this->lbt_m != 1.0) && (ippl::Comm->size() > 1)) {
            mes << "Starting first repartition" << endl;
            this->isFirstRepartition_m           = true;
            this->loadbalancer_m->initializeORB(FL, mesh);
            this->loadbalancer_m->repartition(FL, mesh, this->isFirstRepartition_m);
        }

        static IpplTimings::TimerRef ReadingTimer = IpplTimings::getTimer("Read Data");
        IpplTimings::startTimer(ReadingTimer);

        ifstream file(folder + "Data.csv");

        // Check if the file is opened successfully
        if (!file.is_open()) {
            cerr << "Error opening IC file!" << endl;
        }
        
        // Vector to store data read from the CSV file
        vector<vector<double>> ParticlePositions;
        vector<vector<double>> ParticleVelocities;
        double MaxPos;
        double MinPos;

        // Read the file line by line
        string line;
        while (getline(file, line)) {
            stringstream ss(line);

            // Read each comma-separated value into the row vector
            string cell;
            unsigned int j = 0;
            vector<double> PosRow;
            vector<double> VelRow;
            unsigned int i = 0;
            while (j < 6 && getline(ss, cell, ',')) {
                if (j < 3){
                    double Pos = stod(cell);
                    PosRow.push_back(Pos);
                    // Find Boundaries (x, y, z)
                    if(i > 0){
                        MaxPos = max(Pos, MaxPos);
                        MinPos = min(Pos, MinPos);
                    }
                    else{
                        MaxPos = Pos;
                        MinPos = Pos;
                        ++i;
                    }
                }
                else {
                    double Vel = stod(cell);
                    VelRow.push_back(Vel);
                }
                ++j;
            }
            ParticlePositions.push_back(PosRow);
            ParticleVelocities.push_back(VelRow);
        }

        // Boundaries of Particle Positions
        mes << "Minimum Position: " << MinPos << endl;
        mes << "Maximum Position: " << MaxPos << endl;
        mes << "Defined maximum:  " << this->rmax_m << endl;

        // Number of Particles 
        if (this->totalP_m != ParticlePositions.size()){
            cerr << "Error: Simulation number of particles does not match input!" << endl;
            cerr << "Input N = " << ParticlePositions.size() << ", Simulation N = " << this->totalP_m << endl;
        }    
        else 
            mes << "successfully done." << endl;


        auto R_host = pc->R.getHostMirror();
        auto V_host = pc->V.getHostMirror();

        double a = this->a_m;
        unsigned int j;
        for (unsigned int i = 0; i < nloc; ++i) {
            j = i*ippl::Comm->size() + ippl::Comm->rank();
            R_host(i)[0] = ParticlePositions[j][0];
            R_host(i)[1] = ParticlePositions[j][1];
            R_host(i)[2] = ParticlePositions[j][2];
            V_host(i)[0] = ParticleVelocities[j][0]*pow(a, 1.5);
            V_host(i)[1] = ParticleVelocities[j][1]*pow(a, 1.5);
            V_host(i)[2] = ParticleVelocities[j][2]*pow(a, 1.5);
        }

        Kokkos::deep_copy(pc->R.getView(), R_host);
        Kokkos::deep_copy(pc->V.getView(), V_host);
        Kokkos::fence();
        ippl::Comm->barrier();
        IpplTimings::stopTimer(ReadingTimer);

        // Since the particles have moved spatially update them to correct processors
        pc->update();

        bool isFirstRepartition = false;
        std::shared_ptr<FieldContainer_t> fc    = this->fcontainer_m;
        if (this->loadbalancer_m->balance(this->totalP_m, this->it_m)) {
                auto* mesh = &fc->getRho().get_mesh();
                auto* FL = &fc->getFL();
                this->loadbalancer_m->repartition(FL, mesh, isFirstRepartition);
                printf("first repartition works \n");
        }

        
        mes << "Assignment of positions and velocities done." << endl;

    }

    void advance() override {
        if (this->stepMethod_m == "LeapFrog") {
            LeapFrogStep();
        }
	else{
            throw IpplException(TestName, "Step method is not set/recognized!");
        }
    }

    void LeapFrogStep(){
        // LeapFrog time stepping https://en.wikipedia.org/wiki/Leapfrog_integration

        static IpplTimings::TimerRef VTimer           = IpplTimings::getTimer("pushVelocity");
        static IpplTimings::TimerRef RTimer           = IpplTimings::getTimer("pushPosition");
        static IpplTimings::TimerRef updateTimer      = IpplTimings::getTimer("update");
        static IpplTimings::TimerRef domainDecomposition = IpplTimings::getTimer("loadBalance");
        static IpplTimings::TimerRef SolveTimer       = IpplTimings::getTimer("solve");

        //double dt                               = this->dt_m;
        double a                               = this->a_m;
        double a_i = this->a_m;
        double a_half = a*exp(0.5*this->Dloga);
        double a_f = a*exp(this->Dloga);

        double H_i = this->calculateHubble(a_i);
        double H_half = this->calculateHubble(a_half);
        double H_f = this->calculateHubble(a_f);
        double d_drift, d_kick;

        std::shared_ptr<ParticleContainer_t> pc = this->pcontainer_m;
        std::shared_ptr<FieldContainer_t> fc    = this->fcontainer_m;
        // kick (update V)
        IpplTimings::startTimer(VTimer);
        d_kick = 1./4*(1/(H_i * a_i) + 1/(H_half * a_half))*this->Dloga;
        pc->V = pc->V - 4 * this->G * M_PI * pc->F * d_kick;
        IpplTimings::stopTimer(VTimer);

        // drift (update R) in comoving distances
        IpplTimings::startTimer(RTimer);
        d_drift = 1./6*(1/(H_i * a_i * a_i) + 4/(H_half * a_half * a_half) + 1/(H_f * a_f * a_f))*this->Dloga;
        pc->R = pc->R + pc->V * d_drift;
        IpplTimings::stopTimer(RTimer);

        // Since the particles have moved spatially update them to correct processors
        IpplTimings::startTimer(updateTimer);
        pc->update();
        IpplTimings::stopTimer(updateTimer);
        

        size_type totalP        = this->totalP_m;
        int it                  = this->it_m;
        bool isFirstRepartition = false;
        if (this->loadbalancer_m->balance(totalP, it + 1)) {
                IpplTimings::startTimer(domainDecomposition);
                auto* mesh = &fc->getRho().get_mesh();
                auto* FL = &fc->getFL();
                this->loadbalancer_m->repartition(FL, mesh, isFirstRepartition);
                IpplTimings::stopTimer(domainDecomposition);
        }
        
        // scatter the mass onto the underlying grid
        this->par2grid();

        // Field solve
        IpplTimings::startTimer(SolveTimer);
        this->fsolver_m->runSolver();
        IpplTimings::stopTimer(SolveTimer);

        // gather F field
        this->grid2par();

        // kick (update V)
        IpplTimings::startTimer(VTimer);
        d_kick = 1./4*(1/(H_half * a_half) + 1/(H_f * a_f))*this->Dloga;
        pc->V = pc->V - 4 * this->G * M_PI * pc->F * d_kick;
        IpplTimings::stopTimer(VTimer);

        //if((this->it_m)%100 == 0){
        //    savePositions(this->it_m / 100);
        //}

    }

    void savePositions(unsigned int index) {
        Inform mes("Saving Particles");

        static IpplTimings::TimerRef SavingTimer = IpplTimings::getTimer("Save Data");
        IpplTimings::startTimer(SavingTimer);

        mes << "snapshot " << this->it_m << endl;

        stringstream ss;
        if(ippl::Comm->size() == 1)
            ss << "snapshot_" << std::setfill('0') << std::setw(3) << index;
        else 
            ss << "snapshot_" << ippl::Comm->rank() << "_" << std::setfill('0') << std::setw(3) << index; 
        string filename = ss.str();

        ofstream file(folder + filename + ".csv");

        // Check if the file is opened successfully
        if (!file.is_open()) {
            cerr << "Error opening saving file!" << endl;
            return;
        }
        std::shared_ptr<ParticleContainer_t> pc = this->pcontainer_m;

        auto Rview = this->pcontainer_m->R.getView();
        auto Vview = this->pcontainer_m->V.getView();
        auto Fview = this->pcontainer_m->F.getView();

        auto R_host = this->pcontainer_m->R.getHostMirror();
        auto V_host = this->pcontainer_m->V.getHostMirror();
        auto F_host = this->pcontainer_m->F.getHostMirror();

        Kokkos::deep_copy(R_host, Rview);
        Kokkos::deep_copy(V_host, Vview);
        Kokkos::deep_copy(F_host, Fview);

        double a = this->a_m;


        // Write data to the file
        for (unsigned int i = 0; i < pc->getLocalNum(); ++i){
            for (unsigned int d = 0; d < Dim; ++d)
                file << R_host(i)[d] << ",";
            for (unsigned int d = 0; d < Dim; ++d)
                file << V_host(i)[d] << ",";
            for (unsigned int d = 0; d < Dim; ++d)
                file << - 4*M_PI * this->G / (a*a) * F_host(i)[d] << ",";
            file << "\n";
        }
        ippl::Comm->barrier();

        // Close the file stream
        file.close();
        mes << "done." << endl;
        IpplTimings::stopTimer(SavingTimer);

    }

    /*
    void savePositionsHDF5(unsigned int index) {

        Inform mes("Saving Particles");

        static IpplTimings::TimerRef SavingTimer = IpplTimings::getTimer("Save Data");
        IpplTimings::startTimer(SavingTimer);

        mes << "snapshot " << this->it_m << endl;

        stringstream ss;
        ss << "snapshot_" << std::setfill('0') << std::setw(3) << index;
        string filename = ss.str();

        //ofstream file("data/" + filename + ".csv");
        //ofstream file(folder + filename + ".csv");


        const std::string FILE_NAME = folder + filename + ".h5"; //"data.h5";
        const std::string DATASET_NAME = "particle_data";
   
        auto Rview = this->pcontainer_m->R.getView();
        auto Vview = this->pcontainer_m->V.getView();
        auto Fview = this->pcontainer_m->F.getView();

        auto R_host = this->pcontainer_m->R.getHostMirror();
        auto V_host = this->pcontainer_m->V.getHostMirror();
        auto F_host = this->pcontainer_m->F.getHostMirror();

        Kokkos::deep_copy(R_host, Rview);
        Kokkos::deep_copy(V_host, Vview);
        Kokkos::deep_copy(F_host, Fview);

        //const unsigned int N = 100; // Number of particles
        unsigned int N = Rview.size();
        const unsigned int N_col = 9;

        // Create or open HDF5 file
        hid_t file = H5Fcreate(FILE_NAME.c_str(), H5F_ACC_TRUNC, H5P_DEFAULT, H5P_DEFAULT);
        if (file < 0) {
            std::cerr << "Failed to create HDF5 file." << std::endl;
            //return -1;
        }

        // Create a dataspace for the dataset
        hsize_t dims[2] = {N, N_col}; // Number of rows and columns
        hid_t dataspace = H5Screate_simple(2, dims, NULL);

        // Create the dataset
        hid_t dataset = H5Dcreate2(file, DATASET_NAME.c_str(), H5T_NATIVE_DOUBLE, dataspace, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
        if (dataset < 0) {
            std::cerr << "Failed to create dataset." << std::endl;
            H5Fclose(file);
            //return -1;
        }

        double a = this->a_m;

        // Generate some data to save (e.g., particle data)
        std::vector<double> particleData(N * Dim);
        for (unsigned int i = 0; i < N; ++i) {
                // Populate particle data directly
                particleData[i * Dim] = R_host(i)[0];
                particleData[i * Dim + 1] = R_host(i)[1];
                particleData[i * Dim + 2] = R_host(i)[2];
                particleData[i * Dim + 3] = V_host(i)[0];
                particleData[i * Dim + 4] = V_host(i)[1];
                particleData[i * Dim + 5] = V_host(i)[2];
                particleData[i * Dim + 6] = - 4*M_PI * this->G / (a*a) * F_host(i)[0];
                particleData[i * Dim + 7] = - 4*M_PI * this->G / (a*a) * F_host(i)[1];
                particleData[i * Dim + 8] = - 4*M_PI * this->G / (a*a) * F_host(i)[2];
        }

        // Write data to the dataset
        H5Dwrite(dataset, H5T_NATIVE_DOUBLE, H5S_ALL, H5S_ALL, H5P_DEFAULT, particleData.data());

        // Close resources
        H5Dclose(dataset);
        H5Sclose(dataspace);
        H5Fclose(file);

        // Close the file stream
        mes << "done." << endl;
        IpplTimings::stopTimer(SavingTimer);

    }
    */


    void dump() override {
        static IpplTimings::TimerRef dumpDataTimer = IpplTimings::getTimer("dumpData");
        IpplTimings::startTimer(dumpDataTimer);
        dumpStructure(this->fcontainer_m->getF().getView());
        IpplTimings::stopTimer(dumpDataTimer);
    }

    template <typename View>
    void dumpStructure(const View& Fview) {
        const int nghostF = this->fcontainer_m->getF().getNghost();

        using index_array_type = typename ippl::RangePolicy<Dim>::index_array_type;
        double localEx2 = 0, localExNorm = 0;
        ippl::parallel_reduce(
            "Ex stats", ippl::getRangePolicy(Fview, nghostF),
            KOKKOS_LAMBDA(const index_array_type& args, double& F2, double& FNorm) {
                // ippl::apply<unsigned> accesses the view at the given indices and obtains a
                // reference; see src/Expression/IpplOperations.h
                double val = ippl::apply(Fview, args)[0];
                double f2  = Kokkos::pow(val, 2);
                F2 += f2;

                double norm = Kokkos::fabs(ippl::apply(Fview, args)[0]);
                if (norm > FNorm) {
                    FNorm = norm;
                }
            },
            Kokkos::Sum<double>(localEx2), Kokkos::Max<double>(localExNorm));

        double globaltemp = 0.0;
        ippl::Comm->reduce(localEx2, globaltemp, 1, std::plus<double>());

        double fieldEnergy =
            std::reduce(this->fcontainer_m->getHr().begin(), this->fcontainer_m->getHr().end(), globaltemp, std::multiplies<double>());

        double ExAmp = 0.0;
        ippl::Comm->reduce(localExNorm, ExAmp, 1, std::greater<double>());

        if (ippl::Comm->rank() == 0) {
            std::stringstream fname;
            fname << folder + "FieldStructure_";
            fname << ippl::Comm->size();
            fname << "_manager";
            fname << ".csv";
            Inform csvout(NULL, fname.str().c_str(), Inform::APPEND);
            csvout.precision(16);
            csvout.setf(std::ios::scientific, std::ios::floatfield);
            if ( std::fabs(this->time_m) < 1e-14 ) {
                csvout << "time, Ex_field_energy, Ex_max_norm" << endl;
            }
            csvout << this->time_m << " " << fieldEnergy << " " << ExAmp << endl;
        }
        ippl::Comm->barrier();
    }
};
#endif
